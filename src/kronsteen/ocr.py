"""DeepSeek OCR client wrapper (local HuggingFace model)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from .config import OCRSettings
from .exceptions import OCRConfigurationError, OCRServiceError
from .models import Region, TextMatch


class DeepSeekOCRClient:
    """Loads the open-source DeepSeek OCR model for on-device inference."""

    def __init__(
        self,
        settings: Optional[OCRSettings] = None,
    ) -> None:
        self.settings = settings or OCRSettings.from_env()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._auto_model_cls = None
        self._auto_tokenizer_cls = None

    # region lazy loading --------------------------------------------------------
    def _ensure_dependencies(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - missing deps
            raise OCRConfigurationError(
                "DeepSeek OCR dependencies are missing. Install with `pip install kronsteen[ocr]`. "
                "Note: flash-attn is optional and only available on Linux/Windows with NVIDIA GPU."
            ) from exc
        self._torch = torch
        self._auto_model_cls = AutoModel
        self._auto_tokenizer_cls = AutoTokenizer

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_dependencies()
        assert self._torch is not None
        assert self._auto_model_cls is not None
        assert self._auto_tokenizer_cls is not None

        tokenizer_kwargs = {"trust_remote_code": self.settings.trust_remote_code}
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
            "use_safetensors": True,
        }
        if self.settings.attn_implementation:
            model_kwargs["_attn_implementation"] = self.settings.attn_implementation
        if self.settings.cache_dir:
            tokenizer_kwargs["cache_dir"] = str(self.settings.cache_dir)
            model_kwargs["cache_dir"] = str(self.settings.cache_dir)

        self._tokenizer = self._auto_tokenizer_cls.from_pretrained(
            self.settings.model_name,
            **tokenizer_kwargs,
        )
        self._model = self._auto_model_cls.from_pretrained(
            self.settings.model_name,
            **model_kwargs,
        )
        
        # Auto-detect best available device
        device = self.settings.device
        
        # If CUDA is requested, check if it's available
        if device == "cuda":
            if self._torch.cuda.is_available():
                print("✓ Using CUDA GPU for DeepSeek OCR")
            else:
                device = "cpu"
                print("⚠️  CUDA not available, using CPU for DeepSeek OCR")
                print("   Note: CPU mode is slower. For best performance, use a CUDA-capable GPU.")
        elif device == "cpu":
            # Check if CUDA is available but user chose CPU
            if self._torch.cuda.is_available():
                print("ℹ️  Using CPU for DeepSeek OCR (CUDA available but not selected)")
                print("   Tip: Set DEEPSEEK_DEVICE=cuda for faster performance")
            else:
                print("ℹ️  Using CPU for DeepSeek OCR")
        
        # Auto-detect dtype based on device if not specified
        dtype = None
        if self.settings.torch_dtype:
            dtype = getattr(self._torch, self.settings.torch_dtype, None)
            if dtype is None:
                raise OCRConfigurationError(
                    f"Unknown torch dtype '{self.settings.torch_dtype}'. Set DEEPSEEK_TORCH_DTYPE=none to disable."
                )
        elif device == "cuda":
            # Use bfloat16 on CUDA for better performance
            if hasattr(self._torch, 'bfloat16'):
                dtype = self._torch.bfloat16
                print("   Using bfloat16 precision for CUDA")
        # CPU uses float32 by default (no dtype conversion needed)
        
        if device:
            self._model = self._model.to(device)
        if dtype:
            self._model = self._model.to(dtype)
        self._model = self._model.eval()

    # region public API ----------------------------------------------------------
    def close(self) -> None:
        self._model = None
        self._tokenizer = None

    def extract_text(
        self,
        image: Image.Image,
        *,
        region: Optional[Region] = None,
        prompt: Optional[str] = None,
    ) -> list[TextMatch]:
        self._ensure_model()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        working_image = image.copy()
        offset_x = region.left if region else 0
        offset_y = region.top if region else 0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            working_image.save(temp_path, format="PNG")
        output_dir = self.settings.output_dir or Path(tempfile.mkdtemp(prefix="kronsteen-ocr-"))
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_output_dir = output_dir if self.settings.output_dir else Path(output_dir)

        try:
            # Monkey-patch the model's device for CPU compatibility
            # DeepSeek model hardcodes .cuda() calls, so we need to override
            original_cuda = self._torch.Tensor.cuda
            device_str = str(next(self._model.parameters()).device)
            
            def patched_cuda(tensor_self, *args, **kwargs):
                # If model is on CPU, return self instead of calling .cuda()
                if 'cpu' in device_str:
                    return tensor_self
                return original_cuda(tensor_self, *args, **kwargs)
            
            # Temporarily patch cuda() method
            self._torch.Tensor.cuda = patched_cuda
            
            try:
                infer_kwargs = dict(
                    prompt=prompt or self.settings.prompt,
                    image_file=str(temp_path),
                    output_path=str(generated_output_dir),
                    base_size=self.settings.base_size,
                    image_size=self.settings.image_size,
                    crop_mode=self.settings.crop_mode,
                    save_results=self.settings.save_results,
                    test_compress=self.settings.test_compress,
                )
                result = self._model.infer(self._tokenizer, **infer_kwargs)
            finally:
                # Restore original cuda() method
                self._torch.Tensor.cuda = original_cuda
                
        except Exception as exc:  # pragma: no cover - GPU runtime errors
            raise OCRServiceError(f"DeepSeek inference failed: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

        matches = self._parse_output(result, generated_output_dir, offset_x, offset_y)
        if not self.settings.output_dir:
            shutil.rmtree(generated_output_dir, ignore_errors=True)
        return matches

    # region parsing -------------------------------------------------------------
    def _parse_output(
        self,
        model_output: Any,
        output_dir: Path,
        offset_x: int,
        offset_y: int,
    ) -> list[TextMatch]:
        entries: list[dict[str, Any]] = []
        entries.extend(self._flatten_candidates(model_output))
        entries.extend(self._load_saved_results(output_dir))
        matches: list[TextMatch] = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            region = self._region_from_entry(entry, offset_x, offset_y)
            if not region:
                continue
            confidence = float(entry.get("confidence") or entry.get("score") or entry.get("prob", 1.0))
            matches.append(TextMatch(text=text, confidence=confidence, region=region))
        return matches

    def _flatten_candidates(self, payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            for key in ("data", "results", "ocr", "ocr_info", "grounding", "items"):
                if isinstance(payload.get(key), list):
                    return [entry for entry in payload[key] if isinstance(entry, dict)]
            return [payload]
        return []

    def _load_saved_results(self, output_dir: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for json_file in sorted(output_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
            except Exception:  # pragma: no cover - defensive
                continue
            entries.extend(self._flatten_candidates(data))
        return entries

    def _region_from_entry(self, entry: dict[str, Any], offset_x: int, offset_y: int) -> Optional[Region]:
        box = entry.get("bbox") or entry.get("box") or entry.get("quad") or entry.get("polygon")
        if isinstance(box, Sequence):
            coords = list(map(float, box))
            if len(coords) >= 4:
                if len(coords) == 4:
                    left, top, right, bottom = coords
                else:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)
                return Region(
                    left=int(offset_x + left),
                    top=int(offset_y + top),
                    width=int(max(1, right - left)),
                    height=int(max(1, bottom - top)),
                )
        points = entry.get("points")
        if isinstance(points, list) and points:
            xs: list[float] = []
            ys: list[float] = []
            for point in points:
                if isinstance(point, Sequence) and len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
            if xs and ys:
                left, right = min(xs), max(xs)
                top, bottom = min(ys), max(ys)
                return Region(
                    left=int(offset_x + left),
                    top=int(offset_y + top),
                    width=int(max(1, right - left)),
                    height=int(max(1, bottom - top)),
                )
        return None

    def __enter__(self) -> "DeepSeekOCRClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - convenience
        self.close()
