"""
Computer Vision Module for Kronsteen

Provides object detection, segmentation, and classification capabilities
using YOLO models (via Ultralytics) and Roboflow integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from PIL import Image

from .models import Region

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Detection:
    """Represents a detected object."""
    
    class_name: str
    confidence: float
    region: Region
    class_id: int
    
    @property
    def center(self) -> tuple[int, int]:
        """Get center point of detection."""
        return self.region.center


@dataclass
class Segmentation:
    """Represents a segmented object with mask."""
    
    class_name: str
    confidence: float
    region: Region
    mask: "np.ndarray"  # Binary mask
    class_id: int
    
    @property
    def center(self) -> tuple[int, int]:
        """Get center point of segmentation."""
        return self.region.center


@dataclass
class Classification:
    """Represents a classification result."""
    
    class_name: str
    confidence: float
    class_id: int


class VisionClient:
    """
    Computer vision client for object detection, segmentation, and classification.
    
    Supports:
    - YOLO models (YOLOv8, YOLOv9, YOLOv10, YOLOv11) via Ultralytics
    - Roboflow custom models
    - Pre-trained COCO models
    - Custom trained models
    
    Examples:
        # Object Detection
        >>> vision = VisionClient(model="yolov8n.pt")
        >>> detections = vision.detect(screenshot)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
        
        # Segmentation
        >>> vision = VisionClient(model="yolov8n-seg.pt", task="segment")
        >>> segments = vision.segment(screenshot)
        
        # Classification
        >>> vision = VisionClient(model="yolov8n-cls.pt", task="classify")
        >>> result = vision.classify(screenshot)
    """
    
    def __init__(
        self,
        model: str = "yolov8n.pt",
        task: Literal["detect", "segment", "classify"] = "detect",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        roboflow_api_key: Optional[str] = None,
    ):
        """
        Initialize vision client.
        
        Args:
            model: Model name or path. Options:
                   - "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt" (detection)
                   - "yolov8n-seg.pt" (segmentation)
                   - "yolov8n-cls.pt" (classification)
                   - Path to custom .pt file
                   - Roboflow model ID (e.g., "workspace/project/version")
            task: Task type - "detect", "segment", or "classify"
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device to run on ("cpu", "cuda", "mps", or None for auto)
            roboflow_api_key: Roboflow API key (if using Roboflow models)
        """
        self.model_name = model
        self.task = task
        self.confidence_threshold = confidence_threshold
        self.device = device or self._auto_detect_device()
        self.roboflow_api_key = roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")
        
        self._model: Optional[Any] = None
        self._is_roboflow = "/" in model  # Roboflow models have format: workspace/project/version
        
        print(f"🔍 Initializing {task} model: {model}")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {confidence_threshold}")
    
    def _auto_detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _load_model(self) -> Any:
        """Load the model (lazy loading)."""
        if self._model is not None:
            return self._model
        
        if self._is_roboflow:
            self._model = self._load_roboflow_model()
        else:
            self._model = self._load_ultralytics_model()
        
        return self._model
    
    def _load_ultralytics_model(self) -> Any:
        """Load Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        model = YOLO(self.model_name)
        model.to(self.device)
        return model
    
    def _load_roboflow_model(self) -> Any:
        """Load Roboflow model."""
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow not installed. Install with: pip install roboflow"
            )
        
        if not self.roboflow_api_key:
            raise ValueError(
                "Roboflow API key required. Set ROBOFLOW_API_KEY environment variable "
                "or pass roboflow_api_key parameter."
            )
        
        rf = Roboflow(api_key=self.roboflow_api_key)
        workspace, project, version = self.model_name.split("/")
        project_obj = rf.workspace(workspace).project(project)
        model = project_obj.version(int(version)).model
        return model
    
    def detect(
        self,
        image: Image.Image | str | Path,
        *,
        classes: Optional[list[str]] = None,
        region: Optional[Region] = None,
    ) -> list[Detection]:
        """
        Detect objects in image.
        
        Args:
            image: PIL Image, path to image, or screenshot
            classes: Filter by specific class names (e.g., ["person", "car"])
            region: Region offset for coordinate adjustment
        
        Returns:
            List of Detection objects
        """
        if self.task != "detect":
            raise ValueError(f"Model task is '{self.task}', not 'detect'")
        
        model = self._load_model()
        
        # Convert image to path if needed
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # Save PIL Image temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name
        
        # Run inference
        results = model.predict(
            image_path,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        
        detections: list[Detection] = []
        offset_x = region.left if region else 0
        offset_y = region.top if region else 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # Filter by class if specified
                if classes and cls_name not in classes:
                    continue
                
                # Create detection
                detection = Detection(
                    class_name=cls_name,
                    confidence=conf,
                    region=Region(
                        left=int(x1) + offset_x,
                        top=int(y1) + offset_y,
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    ),
                    class_id=cls_id,
                )
                detections.append(detection)
        
        return detections
    
    def segment(
        self,
        image: Image.Image | str | Path,
        *,
        classes: Optional[list[str]] = None,
        region: Optional[Region] = None,
    ) -> list[Segmentation]:
        """
        Segment objects in image.
        
        Args:
            image: PIL Image, path to image, or screenshot
            classes: Filter by specific class names
            region: Region offset for coordinate adjustment
        
        Returns:
            List of Segmentation objects with masks
        """
        if self.task != "segment":
            raise ValueError(f"Model task is '{self.task}', not 'segment'")
        
        model = self._load_model()
        
        # Convert image to path if needed
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name
        
        # Run inference
        results = model.predict(
            image_path,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        
        segmentations: list[Segmentation] = []
        offset_x = region.left if region else 0
        offset_y = region.top if region else 0
        
        for result in results:
            if result.masks is None:
                continue
            
            boxes = result.boxes
            masks = result.masks
            
            for box, mask in zip(boxes, masks):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # Filter by class if specified
                if classes and cls_name not in classes:
                    continue
                
                # Get mask
                mask_array = mask.data[0].cpu().numpy()
                
                # Create segmentation
                segmentation = Segmentation(
                    class_name=cls_name,
                    confidence=conf,
                    region=Region(
                        left=int(x1) + offset_x,
                        top=int(y1) + offset_y,
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    ),
                    mask=mask_array,
                    class_id=cls_id,
                )
                segmentations.append(segmentation)
        
        return segmentations
    
    def classify(
        self,
        image: Image.Image | str | Path,
        top_k: int = 1,
    ) -> list[Classification]:
        """
        Classify image.
        
        Args:
            image: PIL Image, path to image, or screenshot
            top_k: Return top K predictions
        
        Returns:
            List of Classification objects (sorted by confidence)
        """
        if self.task != "classify":
            raise ValueError(f"Model task is '{self.task}', not 'classify'")
        
        model = self._load_model()
        
        # Convert image to path if needed
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name
        
        # Run inference
        results = model.predict(
            image_path,
            device=self.device,
            verbose=False,
        )
        
        classifications: list[Classification] = []
        
        for result in results:
            probs = result.probs
            top_indices = probs.top5 if top_k >= 5 else probs.top1
            
            for idx in top_indices[:top_k]:
                cls_name = result.names[idx]
                conf = float(probs.data[idx])
                
                classification = Classification(
                    class_name=cls_name,
                    confidence=conf,
                    class_id=idx,
                )
                classifications.append(classification)
        
        return classifications
    
    def find_and_click(
        self,
        image: Image.Image,
        target_class: str,
        *,
        min_confidence: Optional[float] = None,
        click_offset: tuple[int, int] = (0, 0),
    ) -> bool:
        """
        Find object and click on it.
        
        Args:
            image: Screenshot to search in
            target_class: Class name to find (e.g., "button", "person")
            min_confidence: Minimum confidence (overrides default)
            click_offset: Offset from center (x, y)
        
        Returns:
            True if found and clicked, False otherwise
        """
        from . import actions
        
        conf_threshold = min_confidence or self.confidence_threshold
        detections = self.detect(image, classes=[target_class])
        
        # Filter by confidence
        valid_detections = [d for d in detections if d.confidence >= conf_threshold]
        
        if not valid_detections:
            return False
        
        # Click on highest confidence detection
        best_detection = max(valid_detections, key=lambda d: d.confidence)
        center_x, center_y = best_detection.center
        
        actions.click(
            center_x + click_offset[0],
            center_y + click_offset[1],
        )
        
        return True
    
    def close(self) -> None:
        """Close client and free resources."""
        self._model = None
    
    def __enter__(self) -> "VisionClient":
        return self
    
    def __exit__(self, *exc_info: object) -> None:
        self.close()


# Convenience functions for quick access
def detect_objects(
    image: Image.Image,
    model: str = "yolov8n.pt",
    classes: Optional[list[str]] = None,
    confidence: float = 0.25,
) -> list[Detection]:
    """
    Quick object detection.
    
    Example:
        >>> detections = detect_objects(screenshot, classes=["person", "car"])
        >>> for det in detections:
        ...     print(f"{det.class_name} at {det.center}")
    """
    with VisionClient(model=model, task="detect", confidence_threshold=confidence) as client:
        return client.detect(image, classes=classes)


def segment_objects(
    image: Image.Image,
    model: str = "yolov8n-seg.pt",
    classes: Optional[list[str]] = None,
    confidence: float = 0.25,
) -> list[Segmentation]:
    """
    Quick segmentation.
    
    Example:
        >>> segments = segment_objects(screenshot, classes=["person"])
        >>> for seg in segments:
        ...     print(f"{seg.class_name} mask shape: {seg.mask.shape}")
    """
    with VisionClient(model=model, task="segment", confidence_threshold=confidence) as client:
        return client.segment(image, classes=classes)


def classify_image(
    image: Image.Image,
    model: str = "yolov8n-cls.pt",
    top_k: int = 1,
) -> list[Classification]:
    """
    Quick classification.
    
    Example:
        >>> results = classify_image(screenshot, top_k=3)
        >>> for result in results:
        ...     print(f"{result.class_name}: {result.confidence:.2%}")
    """
    with VisionClient(model=model, task="classify") as client:
        return client.classify(image, top_k=top_k)
