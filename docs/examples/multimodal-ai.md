# Multi-Modal AI Service with Modalkit

Deploy a sophisticated multi-modal AI service that processes text, images, and audio using Modalkit, showcasing advanced ML workflows and production deployment patterns.

## Overview

This tutorial demonstrates:
- Multi-modal input processing (text, images, audio)
- Vector embeddings generation
- Semantic search and similarity matching
- Cross-modal understanding
- Production-ready deployment with monitoring
- Advanced queue management for different processing types

## Architecture

```
multimodal-ai/
├── app.py                    # Modal app definition
├── models/
│   ├── text_encoder.py       # Text processing model
│   ├── image_encoder.py      # Image processing model
│   ├── audio_encoder.py      # Audio processing model
│   └── multimodal_model.py   # Combined multi-modal model
├── utils/
│   ├── preprocessing.py      # Input preprocessing utilities
│   ├── embeddings.py         # Vector embedding utilities
│   └── similarity.py         # Similarity computation
├── modalkit.yaml             # Configuration
└── requirements.txt          # Dependencies
```

## 1. Multi-Modal Model Implementation

Create `models/multimodal_model.py`:

```python
from modalkit.inference_pipeline import InferencePipeline
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import whisper
import librosa
import io
import base64
from PIL import Image
import json

# Input schemas
class TextInput(BaseModel):
    text: str
    language: Optional[str] = "en"

class ImageInput(BaseModel):
    image: str  # Base64 encoded
    description: Optional[str] = None

class AudioInput(BaseModel):
    audio: str  # Base64 encoded audio
    sample_rate: Optional[int] = 16000
    format: str = "wav"

class MultiModalInput(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded
    audio: Optional[str] = None  # Base64 encoded
    task: str = "embedding"  # "embedding", "similarity", "search", "classification"
    target_text: Optional[str] = None  # For similarity tasks
    query: Optional[str] = None  # For search tasks
    top_k: int = 5

# Output schemas
class EmbeddingOutput(BaseModel):
    embedding: List[float]
    modality: str
    dimension: int
    model_name: str

class SimilarityOutput(BaseModel):
    similarity_score: float
    modality_pair: str
    method: str

class SearchResult(BaseModel):
    score: float
    content: Dict[str, Any]
    index: int

class MultiModalOutput(BaseModel):
    task: str
    results: List[Dict[str, Any]]
    processing_time: float
    embeddings: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultiModalInference(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = common_settings.get(model_name, {})

        # Initialize models
        self._load_text_model()
        self._load_vision_model()
        self._load_audio_model()

        # Initialize embedding store (in production, use a vector database)
        self.embedding_store = []

        print(f"Multi-modal AI service initialized on {self.device}")

    def _load_text_model(self):
        """Load text embedding model"""
        model_name = self.model_config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")

        if "/mnt/models/text_encoder" in os.listdir("/mnt/models/") if os.path.exists("/mnt/models/") else []:
            print("Loading text model from mounted storage")
            self.text_model = SentenceTransformer("/mnt/models/text_encoder")
        else:
            print(f"Loading text model: {model_name}")
            self.text_model = SentenceTransformer(model_name)

        self.text_model.to(self.device)

    def _load_vision_model(self):
        """Load vision-language model (CLIP)"""
        model_name = self.model_config.get("vision_model", "openai/clip-vit-base-patch32")

        if "/mnt/models/clip_model" in os.listdir("/mnt/models/") if os.path.exists("/mnt/models/") else []:
            print("Loading CLIP model from mounted storage")
            self.clip_model = CLIPModel.from_pretrained("/mnt/models/clip_model")
            self.clip_processor = CLIPProcessor.from_pretrained("/mnt/models/clip_model")
        else:
            print(f"Loading CLIP model: {model_name}")
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        self.clip_model.to(self.device)

    def _load_audio_model(self):
        """Load audio processing model (Whisper)"""
        model_size = self.model_config.get("audio_model", "base")

        if f"/mnt/models/whisper_{model_size}" in os.listdir("/mnt/models/") if os.path.exists("/mnt/models/") else []:
            print("Loading Whisper model from mounted storage")
            self.whisper_model = whisper.load_model(f"/mnt/models/whisper_{model_size}")
        else:
            print(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)

    def _decode_image(self, image_b64: str) -> Image.Image:
        """Decode base64 image"""
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes))

    def _decode_audio(self, audio_b64: str, sample_rate: int = 16000) -> np.ndarray:
        """Decode base64 audio"""
        audio_bytes = base64.b64decode(audio_b64)
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate)
        return audio_array

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding"""
        return self.text_model.encode(text, convert_to_tensor=True).cpu().numpy()

    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding using CLIP"""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.cpu().numpy()[0]

    def _get_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Generate audio embedding by transcribing and then encoding text"""
        # Transcribe audio
        result = self.whisper_model.transcribe(audio)
        text = result["text"]

        # Generate embedding from transcription
        embedding = self._get_text_embedding(text)

        return embedding, text

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def preprocess(self, input_list: List[MultiModalInput]) -> dict:
        """Preprocess multi-modal inputs"""
        import time
        start_time = time.time()

        text_inputs = []
        image_inputs = []
        audio_inputs = []
        tasks = []

        for input_item in input_list:
            tasks.append(input_item.task)

            if input_item.text:
                text_inputs.append(input_item.text)
            else:
                text_inputs.append(None)

            if input_item.image:
                image = self._decode_image(input_item.image)
                image_inputs.append(image)
            else:
                image_inputs.append(None)

            if input_item.audio:
                audio = self._decode_audio(input_item.audio)
                audio_inputs.append(audio)
            else:
                audio_inputs.append(None)

        preprocessing_time = time.time() - start_time

        return {
            "text_inputs": text_inputs,
            "image_inputs": image_inputs,
            "audio_inputs": audio_inputs,
            "tasks": tasks,
            "input_items": input_list,
            "preprocessing_time": preprocessing_time
        }

    def predict(self, input_list: List[MultiModalInput], preprocessed_data: dict) -> dict:
        """Process multi-modal inputs based on task type"""
        import time
        start_time = time.time()

        results = []
        all_embeddings = {}

        for i, (input_item, text, image, audio) in enumerate(zip(
            input_list,
            preprocessed_data["text_inputs"],
            preprocessed_data["image_inputs"],
            preprocessed_data["audio_inputs"]
        )):
            item_result = {"index": i}
            item_embeddings = {}

            # Generate embeddings for available modalities
            if text:
                text_emb = self._get_text_embedding(text)
                item_embeddings["text"] = text_emb.tolist()

            if image is not None:
                image_emb = self._get_image_embedding(image)
                item_embeddings["image"] = image_emb.tolist()

            if audio is not None:
                audio_emb, transcription = self._get_audio_embedding(audio)
                item_embeddings["audio"] = audio_emb.tolist()
                item_result["transcription"] = transcription

            # Process based on task type
            if input_item.task == "embedding":
                item_result["embeddings"] = item_embeddings
                item_result["dimensions"] = {k: len(v) for k, v in item_embeddings.items()}

            elif input_item.task == "similarity":
                if input_item.target_text and text:
                    target_emb = self._get_text_embedding(input_item.target_text)
                    text_emb = np.array(item_embeddings["text"])
                    similarity = self._compute_similarity(text_emb, target_emb)
                    item_result["similarity_score"] = float(similarity)
                    item_result["modality_pair"] = "text-text"

                elif input_item.target_text and image is not None:
                    # Text-image similarity using CLIP
                    inputs = self.clip_processor(
                        text=[input_item.target_text],
                        images=image,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)

                    item_result["similarity_score"] = float(probs[0][0])
                    item_result["modality_pair"] = "image-text"

            elif input_item.task == "search":
                if input_item.query and item_embeddings:
                    query_emb = self._get_text_embedding(input_item.query)

                    # Search in embedding store (simplified)
                    search_results = []
                    for idx, stored_item in enumerate(self.embedding_store):
                        for modality, stored_emb in stored_item["embeddings"].items():
                            if modality in item_embeddings:
                                similarity = self._compute_similarity(
                                    query_emb, np.array(stored_emb)
                                )
                                search_results.append({
                                    "score": float(similarity),
                                    "index": idx,
                                    "modality": modality,
                                    "content": stored_item.get("content", {})
                                })

                    # Sort by similarity and return top-k
                    search_results.sort(key=lambda x: x["score"], reverse=True)
                    item_result["search_results"] = search_results[:input_item.top_k]

            elif input_item.task == "classification":
                # Multi-modal classification (simplified example)
                if text and image is not None:
                    # Combine text and image features for classification
                    combined_features = np.concatenate([
                        np.array(item_embeddings["text"]),
                        np.array(item_embeddings["image"])
                    ])

                    # Dummy classification (replace with actual classifier)
                    categories = ["positive", "negative", "neutral"]
                    scores = np.random.softmax(np.random.randn(len(categories)))

                    item_result["classification"] = [
                        {"category": cat, "score": float(score)}
                        for cat, score in zip(categories, scores)
                    ]

            # Store embeddings for future search
            if input_item.task == "embedding":
                self.embedding_store.append({
                    "embeddings": item_embeddings,
                    "content": {
                        "text": text,
                        "has_image": image is not None,
                        "has_audio": audio is not None
                    }
                })

            results.append(item_result)
            all_embeddings[f"item_{i}"] = item_embeddings

        inference_time = time.time() - start_time

        return {
            "results": results,
            "all_embeddings": all_embeddings,
            "inference_time": inference_time,
            "total_items": len(input_list)
        }

    def postprocess(self, input_list: List[MultiModalInput], raw_output: dict) -> List[MultiModalOutput]:
        """Format outputs with metadata"""
        outputs = []

        for i, (input_item, result) in enumerate(zip(input_list, raw_output["results"])):
            metadata = {
                "processing_time": raw_output["inference_time"],
                "modalities_processed": [],
                "model_versions": {
                    "text_model": getattr(self.text_model, "model_name", "unknown"),
                    "vision_model": "clip-vit-base-patch32",
                    "audio_model": "whisper-base"
                }
            }

            if input_item.text:
                metadata["modalities_processed"].append("text")
            if input_item.image:
                metadata["modalities_processed"].append("image")
            if input_item.audio:
                metadata["modalities_processed"].append("audio")

            outputs.append(MultiModalOutput(
                task=input_item.task,
                results=[result],
                processing_time=raw_output["inference_time"],
                embeddings=raw_output["all_embeddings"].get(f"item_{i}"),
                metadata=metadata
            ))

        return outputs
```

## 2. Configuration

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "multimodal-ai"

  # Authentication
  auth_config:
    ssm_key: "/multimodal-ai/api-key"
    auth_header: "x-api-key"

  # Container with ML dependencies
  build_config:
    image: "python:3.11"
    tag: "latest"
    workdir: "/app"
    env:
      TRANSFORMERS_CACHE: "/tmp/transformers_cache"
      SENTENCE_TRANSFORMERS_HOME: "/tmp/sentence_transformers"
      WHISPER_CACHE: "/tmp/whisper_cache"
    extra_run_commands:
      # System dependencies
      - "apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1 libsndfile1"
      # PyTorch with CUDA
      - "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
      # ML libraries
      - "pip install transformers sentence-transformers"
      - "pip install openai-whisper librosa soundfile"
      - "pip install pillow opencv-python"
      # Vector database (optional)
      - "pip install faiss-cpu chromadb"

  # High-memory GPU deployment
  deployment_config:
    gpu: "A10G"  # Need more VRAM for multi-modal models
    concurrency_limit: 4
    container_idle_timeout: 900  # 15 minutes
    retries: 3
    memory: 32768  # 32GB RAM for large models

    # Mount pre-trained models
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "multimodal-models"
        secret: "aws-credentials"
        key_prefix: "production/"
        read_only: true

    # Cache volumes
    volumes:
      "/tmp/transformers_cache": "transformers-cache"
      "/tmp/sentence_transformers": "sentence-transformers-cache"
      "/tmp/whisper_cache": "whisper-cache"
    volume_reload_interval_seconds: 3600

  # Batch processing for multiple modalities
  batch_config:
    max_batch_size: 4  # Smaller batches due to memory requirements
    wait_ms: 200

  # Queue for different processing types
  queue_config:
    backend: "taskiq"
    broker_url: "redis://redis:6379"

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    device: "cuda"
    precision: "float16"
  model_entries:
    multimodal_model:
      text_model: "sentence-transformers/all-MiniLM-L6-v2"
      vision_model: "openai/clip-vit-base-patch32"
      audio_model: "base"  # Whisper model size
      embedding_dim: 384
```

## 3. Modal App

Create `app.py`:

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig
from models.multimodal_model import MultiModalInference, MultiModalInput, MultiModalOutput

# Initialize Modalkit
modal_utils = ModalConfig()
app = modal.App(name=modal_utils.app_name)

# Define Modal app class
@app.cls(**modal_utils.get_app_cls_settings())
class MultiModalApp(ModalService):
    inference_implementation = MultiModalInference
    model_name: str = modal.parameter(default="multimodal_model")
    modal_utils: ModalConfig = modal_utils

# Create endpoints
@app.function(**modal_utils.get_handler_settings())
@modal.asgi_app(**modal_utils.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=MultiModalApp,
        input_model=MultiModalInput,
        output_model=MultiModalOutput
    )

# Specialized endpoints for different tasks
@app.function(**modal_utils.get_handler_settings())
def generate_embeddings(content: dict):
    """Generate embeddings for content indexing"""
    # This could be used for batch processing
    pass

@app.function(**modal_utils.get_handler_settings())
def similarity_search(query: str, modality: str = "text"):
    """Perform similarity search across modalities"""
    # This could be used for search functionality
    pass

if __name__ == "__main__":
    with modal.enable_local_development():
        pass
```

## 4. Usage Examples

### Text-to-Image Similarity

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Check if image matches text description
headers = {"x-api-key": "your-api-key"}
image_b64 = encode_image("path/to/image.jpg")

response = requests.post(
    "https://your-org--multimodal-ai.modal.run/predict_sync",
    json={
        "image": image_b64,
        "target_text": "a red car parked in front of a house",
        "task": "similarity"
    },
    headers=headers
)

result = response.json()
print(f"Similarity score: {result['results'][0]['similarity_score']:.3f}")
```

### Audio Transcription and Embedding

```python
import requests
import base64
import wave

def encode_audio(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Process audio file
headers = {"x-api-key": "your-api-key"}
audio_b64 = encode_audio("path/to/audio.wav")

response = requests.post(
    "https://your-org--multimodal-ai.modal.run/predict_sync",
    json={
        "audio": audio_b64,
        "task": "embedding"
    },
    headers=headers
)

result = response.json()
print(f"Transcription: {result['results'][0]['transcription']}")
print(f"Audio embedding dimension: {result['results'][0]['dimensions']['audio']}")
```

### Multi-Modal Content Search

```python
import requests

# Search for content using text query
headers = {"x-api-key": "your-api-key"}

response = requests.post(
    "https://your-org--multimodal-ai.modal.run/predict_sync",
    json={
        "text": "example content to index",
        "query": "find similar content",
        "task": "search",
        "top_k": 5
    },
    headers=headers
)

results = response.json()
for result in results['results'][0]['search_results']:
    print(f"Score: {result['score']:.3f} - {result['content']}")
```

### Batch Multi-Modal Processing

```python
import requests
import base64

# Process multiple items with different modalities
headers = {"x-api-key": "your-api-key"}

batch_data = [
    {
        "text": "A beautiful sunset over the ocean",
        "task": "embedding"
    },
    {
        "image": encode_image("image1.jpg"),
        "text": "beach scene",
        "task": "similarity"
    },
    {
        "audio": encode_audio("audio1.wav"),
        "task": "embedding"
    }
]

response = requests.post(
    "https://your-org--multimodal-ai.modal.run/predict_batch",
    json=batch_data,
    headers=headers
)

results = response.json()
for i, result in enumerate(results):
    print(f"Item {i+1}: {result['task']} - Processed {len(result['metadata']['modalities_processed'])} modalities")
```

## 5. Advanced Features

### Vector Database Integration

```python
import chromadb
from chromadb.config import Settings

class VectorStoreInference(MultiModalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize vector database
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="/tmp/chroma_db",
            anonymized_telemetry=False
        ))

        self.collection = self.chroma_client.get_or_create_collection(
            name="multimodal_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def _store_embedding(self, embedding, metadata, item_id):
        """Store embedding in vector database"""
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[item_id]
        )

    def _search_similar(self, query_embedding, top_k=5):
        """Search for similar embeddings"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results
```

### Real-time Processing

```python
import asyncio
from modalkit.task_queue import QueueBackend

class StreamingMultiModalInference(MultiModalInference):
    async def process_stream(self, stream_data):
        """Process streaming multi-modal data"""
        # Process data in chunks
        for chunk in stream_data:
            # Process each chunk
            result = await self.process_chunk(chunk)
            yield result

    async def process_chunk(self, chunk):
        """Process individual chunk"""
        # Implement streaming processing logic
        pass
```

### Cross-Modal Generation

```python
from diffusers import StableDiffusionPipeline

class GenerativeMultiModalInference(MultiModalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load generative models
        self.text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(self.device)

    def generate_image_from_text(self, text_prompt):
        """Generate image from text description"""
        with torch.no_grad():
            image = self.text_to_image_pipe(text_prompt).images[0]
        return image
```

## 6. Production Deployment

### High-Availability Configuration

```yaml
deployment_config:
  gpu: "A100"  # High-end GPU for production
  concurrency_limit: 8
  container_idle_timeout: 1800  # 30 minutes

  # Multiple availability zones
  region: "us-east-1"

  # Health check configuration
  health_check_path: "/health"
  health_check_interval: 30
```

### Cost Optimization

```yaml
# Use spot instances for non-critical workloads
deployment_config:
  gpu: "T4"  # Cost-effective option
  concurrency_limit: 16
  container_idle_timeout: 300  # Scale down quickly

  # Preemptible instances
  preemptible: true
```

### Monitoring and Observability

```python
import time
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('multimodal_requests_total', 'Total requests', ['modality', 'task'])
REQUEST_DURATION = Histogram('multimodal_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('multimodal_active_connections', 'Active connections')

class MonitoredMultiModalInference(MultiModalInference):
    def predict(self, input_list, preprocessed_data):
        start_time = time.time()

        # Track metrics
        for input_item in input_list:
            modalities = []
            if input_item.text:
                modalities.append("text")
            if input_item.image:
                modalities.append("image")
            if input_item.audio:
                modalities.append("audio")

            for modality in modalities:
                REQUEST_COUNT.labels(modality=modality, task=input_item.task).inc()

        result = super().predict(input_list, preprocessed_data)

        # Record duration
        REQUEST_DURATION.observe(time.time() - start_time)

        return result
```

## Key Features Demonstrated

1. **Multi-Modal Processing**: Unified handling of text, images, and audio
2. **Vector Embeddings**: Generate and compare embeddings across modalities
3. **Semantic Search**: Cross-modal similarity and search capabilities
4. **Batch Processing**: Efficient processing of multiple modalities
5. **Production Ready**: Comprehensive monitoring and error handling
6. **Scalable Architecture**: Configurable for different workload requirements
7. **Advanced ML**: Integration of state-of-the-art models (CLIP, Whisper, transformers)

This example showcases Modalkit's capability to handle complex, multi-modal AI workloads with production-grade features and performance optimization.
