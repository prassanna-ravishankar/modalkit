# Computer Vision with Modalkit

Deploy a computer vision model for image classification and object detection using Modalkit, demonstrating GPU acceleration, cloud storage, and production deployment patterns.

## Overview

This tutorial covers:
- Image classification with pre-trained models
- Object detection with YOLO/DETR
- Image preprocessing and augmentation
- Cloud storage for model artifacts
- Batch processing of images
- Production deployment with monitoring

## Project Structure

```
cv-service/
├── app.py                 # Modal app definition
├── vision_model.py        # Computer vision inference
├── utils.py               # Image processing utilities
├── modalkit.yaml          # Configuration
├── requirements.txt       # Dependencies
└── models/                # Model artifacts
    ├── classifier.pth
    └── detector.pth
```

## 1. Vision Model Implementation

Create `vision_model.py`:

```python
from modalkit.inference import InferencePipeline
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import io
import base64
import json
import os

# Input/Output schemas
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    task: str = "classification"  # "classification" or "detection"
    confidence_threshold: float = 0.5
    max_detections: int = 10

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class ImageOutput(BaseModel):
    task: str
    predictions: List[Dict[str, Any]]
    processing_time: float
    image_size: tuple
    model_name: str

class ComputerVisionInference(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)

        self.model_config = common_settings.get(model_name, {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classification model
        self.classification_model = self._load_classification_model()

        # Load detection model (optional)
        self.detection_model = self._load_detection_model()

        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load class labels
        self.class_labels = self._load_class_labels()

        print(f"Computer Vision model loaded on {self.device}")

    def _load_classification_model(self):
        """Load classification model from cloud storage or download"""
        model_path = "/mnt/models/classifier.pth"

        if os.path.exists(model_path):
            print("Loading classification model from mounted storage")
            model = resnet50(weights=None)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Loading pre-trained ResNet50 model")
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        model.to(self.device)
        model.eval()
        return model

    def _load_detection_model(self):
        """Load object detection model (YOLO/DETR)"""
        try:
            import ultralytics
            model_path = "/mnt/models/yolov8n.pt"

            if os.path.exists(model_path):
                print("Loading YOLO model from mounted storage")
                model = ultralytics.YOLO(model_path)
            else:
                print("Loading pre-trained YOLOv8 model")
                model = ultralytics.YOLO('yolov8n.pt')

            return model
        except ImportError:
            print("YOLO not available, detection disabled")
            return None

    def _load_class_labels(self):
        """Load class labels from file or use ImageNet labels"""
        labels_path = "/mnt/models/class_labels.json"

        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                return json.load(f)
        else:
            # Use ImageNet labels
            return self._get_imagenet_labels()

    def _get_imagenet_labels(self):
        """Get ImageNet class labels"""
        # This would typically be loaded from a file
        # For brevity, returning a subset
        return {
            0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
            4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock",
            8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch",
            # ... (1000 total classes)
        }

    def _decode_image(self, image_b64: str) -> Image.Image:
        """Decode base64 image"""
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    def preprocess(self, input_list: List[ImageInput]) -> dict:
        """Preprocess images for batch inference"""
        import time
        start_time = time.time()

        images = []
        original_sizes = []
        tasks = []

        for input_item in input_list:
            # Decode image
            image = self._decode_image(input_item.image)
            original_sizes.append(image.size)

            # Transform for classification
            if input_item.task == "classification":
                transformed = self.transform(image)
                images.append(transformed)
            else:
                # For detection, keep original image
                images.append(np.array(image))

            tasks.append(input_item.task)

        # Stack classification images into batch tensor
        classification_images = [img for img, task in zip(images, tasks) if task == "classification"]
        if classification_images:
            classification_batch = torch.stack(classification_images).to(self.device)
        else:
            classification_batch = None

        # Keep detection images as list
        detection_images = [img for img, task in zip(images, tasks) if task == "detection"]

        preprocessing_time = time.time() - start_time

        return {
            "classification_batch": classification_batch,
            "detection_images": detection_images,
            "original_sizes": original_sizes,
            "tasks": tasks,
            "preprocessing_time": preprocessing_time
        }

    def predict(self, input_list: List[ImageInput], preprocessed_data: dict) -> dict:
        """Run inference on preprocessed images"""
        import time
        start_time = time.time()

        classification_results = []
        detection_results = []

        # Classification inference
        if preprocessed_data["classification_batch"] is not None:
            with torch.no_grad():
                outputs = self.classification_model(preprocessed_data["classification_batch"])
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                for probs in probabilities:
                    # Get top-5 predictions
                    top5_prob, top5_indices = torch.topk(probs, 5)
                    predictions = [
                        {
                            "class_id": int(idx),
                            "class_name": self.class_labels.get(int(idx), f"class_{idx}"),
                            "confidence": float(prob)
                        }
                        for idx, prob in zip(top5_indices, top5_prob)
                    ]
                    classification_results.append(predictions)

        # Detection inference
        if preprocessed_data["detection_images"] and self.detection_model:
            for image in preprocessed_data["detection_images"]:
                results = self.detection_model(image)

                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])

                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": self.detection_model.names[class_id]
                            })

                detection_results.append(detections)

        inference_time = time.time() - start_time

        return {
            "classification_results": classification_results,
            "detection_results": detection_results,
            "inference_time": inference_time
        }

    def postprocess(self, input_list: List[ImageInput], raw_output: dict) -> List[ImageOutput]:
        """Format outputs with metadata"""
        outputs = []

        classification_idx = 0
        detection_idx = 0

        for i, input_item in enumerate(input_list):
            if input_item.task == "classification":
                predictions = raw_output["classification_results"][classification_idx]
                # Filter by confidence threshold
                filtered_predictions = [
                    pred for pred in predictions
                    if pred["confidence"] >= input_item.confidence_threshold
                ]
                classification_idx += 1
            else:  # detection
                predictions = raw_output["detection_results"][detection_idx] if detection_idx < len(raw_output["detection_results"]) else []
                # Filter by confidence threshold and max detections
                filtered_predictions = [
                    pred for pred in predictions
                    if pred["confidence"] >= input_item.confidence_threshold
                ][:input_item.max_detections]
                detection_idx += 1

            outputs.append(ImageOutput(
                task=input_item.task,
                predictions=filtered_predictions,
                processing_time=raw_output["inference_time"],
                image_size=raw_output.get("original_sizes", [(0, 0)])[i],
                model_name=self.model_config.get("model_name", "vision_model")
            ))

        return outputs
```

## 2. Configuration

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "cv-service"

  # Authentication
  auth_config:
    ssm_key: "/cv-service/api-key"
    auth_header: "x-api-key"

  # Container with CV dependencies
  build_config:
    image: "python:3.11"
    tag: "latest"
    workdir: "/app"
    env:
      OPENCV_VERSION: "4.8.0"
      TORCH_VERSION: "2.0.0"
    extra_run_commands:
      # Install system dependencies
      - "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1"
      # Install PyTorch with CUDA support
      - "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
      # Install OpenCV and other CV libraries
      - "pip install opencv-python pillow ultralytics"
      # Install additional ML libraries
      - "pip install scikit-image matplotlib seaborn"

  # GPU deployment for CV workloads
  deployment_config:
    gpu: "T4"  # T4 is good for most CV tasks, A10G for larger models
    concurrency_limit: 8
    container_idle_timeout: 600
    retries: 3
    memory: 16384  # 16GB RAM for image processing

    # Mount models and datasets from cloud storage
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "cv-models-bucket"
        secret: "aws-credentials"
        key_prefix: "production/"
        read_only: true
      - mount_point: "/mnt/datasets"
        bucket_name: "cv-datasets-bucket"
        secret: "aws-credentials"
        key_prefix: "validation/"
        read_only: true

    # Cache for model downloads
    volumes:
      "/tmp/model_cache": "cv-model-cache"
    volume_reload_interval_seconds: 1800  # 30 minutes

  # Batch processing for multiple images
  batch_config:
    max_batch_size: 16  # Process multiple images efficiently
    wait_ms: 150

  # Async processing for large images
  queue_config:
    backend: "taskiq"
    broker_url: "redis://redis:6379"

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    device: "cuda"
    model_cache_dir: "/tmp/model_cache"
  model_entries:
    cv_model:
      model_name: "resnet50_cv"
      num_classes: 1000
      input_size: 224
    detection_model:
      model_name: "yolov8n"
      confidence_threshold: 0.5
      iou_threshold: 0.45
```

## 3. Modal App

Create `app.py`:

```python
import modal
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from vision_model import ComputerVisionInference, ImageInput, ImageOutput

# Initialize Modalkit
modal_utils = ModalConfig()
app = modal.App(name=modal_utils.app_name)

# Define Modal app class
@app.cls(**modal_utils.get_app_cls_settings())
class CVApp(ModalService):
    inference_implementation = ComputerVisionInference
    model_name: str = modal.parameter(default="cv_model")
    modal_utils: ModalConfig = modal_utils

# Create endpoints
@app.function(**modal_utils.get_handler_settings())
@modal.asgi_app(**modal_utils.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=CVApp,
        input_model=ImageInput,
        output_model=ImageOutput
    )

# Utility function for image processing
@app.function(
    gpu="T4",
    image=modal.Image.debian_slim().pip_install(
        "opencv-python", "pillow", "torch", "torchvision"
    )
)
def process_image_batch(image_paths: list):
    """Process a batch of images from cloud storage"""
    # This could be used for batch processing from S3/GCS
    pass

if __name__ == "__main__":
    with modal.enable_local_development():
        pass
```

## 4. Usage Examples

### Image Classification

```python
import requests
import base64
from PIL import Image
import io

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Classify an image
headers = {"x-api-key": "your-api-key"}
image_b64 = encode_image("path/to/image.jpg")

response = requests.post(
    "https://your-org--cv-service.modal.run/predict_sync",
    json={
        "image": image_b64,
        "task": "classification",
        "confidence_threshold": 0.1
    },
    headers=headers
)

result = response.json()
print("Top predictions:")
for pred in result["predictions"][:3]:
    print(f"  {pred['class_name']}: {pred['confidence']:.3f}")
```

### Object Detection

```python
import requests
import base64

# Detect objects in an image
headers = {"x-api-key": "your-api-key"}
image_b64 = encode_image("path/to/image.jpg")

response = requests.post(
    "https://your-org--cv-service.modal.run/predict_sync",
    json={
        "image": image_b64,
        "task": "detection",
        "confidence_threshold": 0.5,
        "max_detections": 10
    },
    headers=headers
)

result = response.json()
print("Detected objects:")
for detection in result["predictions"]:
    bbox = detection["bbox"]
    print(f"  {detection['class_name']}: {detection['confidence']:.3f} at [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
```

### Batch Processing

```python
import requests
import base64
import os

# Process multiple images
headers = {"x-api-key": "your-api-key"}
image_folder = "path/to/images/"

batch_requests = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image_b64 = encode_image(image_path)
        batch_requests.append({
            "image": image_b64,
            "task": "classification",
            "confidence_threshold": 0.2
        })

response = requests.post(
    "https://your-org--cv-service.modal.run/predict_batch",
    json=batch_requests,
    headers=headers
)

results = response.json()
for i, result in enumerate(results):
    print(f"Image {i+1}: {result['predictions'][0]['class_name']} ({result['predictions'][0]['confidence']:.3f})")
```

## 5. Advanced Features

### Custom Model Loading

```python
class CustomVisionInference(ComputerVisionInference):
    def _load_classification_model(self):
        """Load custom trained model"""
        import timm

        model_path = "/mnt/models/custom_classifier.pth"
        if os.path.exists(model_path):
            # Load custom model
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Fallback to pre-trained
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1000)

        model.to(self.device)
        model.eval()
        return model
```

### Image Augmentation

```python
def get_augmentation_transforms(self):
    """Get data augmentation transforms"""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

### Performance Monitoring

```python
import time
import logging

logger = logging.getLogger(__name__)

class MonitoredVisionInference(ComputerVisionInference):
    def predict(self, input_list, preprocessed_data):
        start_time = time.time()

        # Log input characteristics
        logger.info(f"Processing {len(input_list)} images")
        logger.info(f"Tasks: {preprocessed_data['tasks']}")

        result = super().predict(input_list, preprocessed_data)

        # Log performance metrics
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f}s")
        logger.info(f"Average time per image: {inference_time/len(input_list):.3f}s")

        return result
```

## 6. Production Deployment

### Multi-GPU Configuration

```yaml
deployment_config:
  gpu: "A10G"  # More powerful GPU for large models
  concurrency_limit: 4  # Lower limit for GPU memory

batch_config:
  max_batch_size: 8  # Smaller batches for GPU memory constraints
```

### Auto-scaling Configuration

```yaml
deployment_config:
  concurrency_limit: 20
  container_idle_timeout: 300  # Scale down quickly

  # Use cheaper instances for light workloads
  cpu: 8.0
  memory: 32768
```

### Model Versioning

```yaml
model_settings:
  model_entries:
    cv_model_v1:
      model_name: "resnet50"
      checkpoint: "v1.0"
    cv_model_v2:
      model_name: "efficientnet_b3"
      checkpoint: "v2.0"
```

## 7. Error Handling and Validation

```python
from modalkit.exceptions import DependencyError

def validate_image_input(self, image_b64: str) -> None:
    """Validate image input"""
    try:
        # Check if it's valid base64
        image_bytes = base64.b64decode(image_b64)

        # Check file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("Image too large (max 10MB)")

        # Check if it's a valid image
        image = Image.open(io.BytesIO(image_bytes))

        # Check dimensions
        if image.width * image.height > 10000 * 10000:
            raise ValueError("Image dimensions too large")

    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")
```

## Key Features Demonstrated

1. **Multi-Modal Support**: Both classification and detection in one service
2. **GPU Acceleration**: Efficient GPU usage for computer vision workloads
3. **Batch Processing**: Process multiple images efficiently
4. **Cloud Storage**: Model and dataset loading from S3/GCS
5. **Production Ready**: Proper error handling, monitoring, and scaling
6. **Flexible Input**: Support for various image formats and sizes
7. **Model Versioning**: Easy model updates and A/B testing

This example shows how Modalkit simplifies deploying complex computer vision models with production-grade features and performance optimization.
