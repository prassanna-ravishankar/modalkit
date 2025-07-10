# Sentiment Analysis Service

A production-ready sentiment analysis service using Modalkit, demonstrating core features with a practical use case.

## Overview

This tutorial shows how to build a sentiment analysis API that:
- Analyzes text sentiment using pre-trained models
- Provides confidence scores and emotion detection
- Handles batch processing for multiple texts
- Includes proper error handling and monitoring
- Demonstrates cloud storage integration

## Project Structure

```
sentiment-service/
├── app.py                # Modal app definition
├── sentiment_model.py    # Sentiment analysis implementation
├── modalkit.yaml         # Configuration
├── requirements.txt      # Dependencies
└── models/               # Model artifacts (optional)
```

## 1. Model Implementation

Create `sentiment_model.py`:

```python
from modalkit.inference import InferencePipeline
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# Input/Output schemas
class TextInput(BaseModel):
    text: str
    language: str = "en"
    include_emotions: bool = False

class SentimentOutput(BaseModel):
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    score: float  # -1 to 1 scale
    emotions: Optional[Dict[str, float]] = None
    language: str
    processing_time: float

class SentimentAnalysisInference(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)

        self.model_config = common_settings.get(model_name, {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize sentiment analysis pipeline
        self.sentiment_pipeline = self._load_sentiment_model()

        # Initialize emotion detection (optional)
        self.emotion_pipeline = self._load_emotion_model()

        # Language detection (for multilingual support)
        self.language_detector = self._load_language_detector()

        print(f"Sentiment analysis service initialized on {self.device}")

    def _load_sentiment_model(self):
        """Load sentiment analysis model"""
        model_name = self.model_config.get("sentiment_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")

        # Try to load from mounted storage first
        model_path = "/mnt/models/sentiment_model"
        if os.path.exists(model_path):
            print("Loading sentiment model from mounted storage")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        else:
            print(f"Loading sentiment model from Hugging Face: {model_name}")
            return pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def _load_emotion_model(self):
        """Load emotion detection model"""
        try:
            model_name = self.model_config.get("emotion_model", "j-hartmann/emotion-english-distilroberta-base")
            emotion_path = "/mnt/models/emotion_model"

            if os.path.exists(emotion_path):
                print("Loading emotion model from mounted storage")
                tokenizer = AutoTokenizer.from_pretrained(emotion_path)
                model = AutoModelForSequenceClassification.from_pretrained(emotion_path)
                return pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            else:
                print(f"Loading emotion model from Hugging Face: {model_name}")
                return pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            return None

    def _load_language_detector(self):
        """Load language detection model"""
        try:
            from langdetect import detect
            return detect
        except ImportError:
            print("Language detection not available")
            return None

    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        if self.language_detector:
            try:
                return self.language_detector(text)
            except:
                return "unknown"
        return "unknown"

    def _normalize_sentiment_score(self, label: str, score: float) -> tuple[str, float, float]:
        """Normalize sentiment score to -1 to 1 scale"""
        if label.lower() in ["positive", "pos"]:
            return "positive", score, score
        elif label.lower() in ["negative", "neg"]:
            return "negative", score, -score
        else:
            return "neutral", score, 0.0

    def _get_emotions(self, text: str) -> Optional[Dict[str, float]]:
        """Get emotion scores for text"""
        if not self.emotion_pipeline:
            return None

        try:
            results = self.emotion_pipeline(text)
            emotions = {}
            for result in results:
                emotions[result['label'].lower()] = result['score']
            return emotions
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None

    def preprocess(self, input_list: List[TextInput]) -> dict:
        """Preprocess text inputs for sentiment analysis"""
        import time
        start_time = time.time()

        texts = []
        languages = []
        include_emotions_flags = []

        for input_item in input_list:
            # Clean and prepare text
            cleaned_text = input_item.text.strip()
            if len(cleaned_text) == 0:
                cleaned_text = "empty text"

            # Truncate very long texts
            if len(cleaned_text) > 512:
                cleaned_text = cleaned_text[:512]

            texts.append(cleaned_text)

            # Detect language if not provided
            if input_item.language == "auto":
                detected_lang = self._detect_language(cleaned_text)
                languages.append(detected_lang)
            else:
                languages.append(input_item.language)

            include_emotions_flags.append(input_item.include_emotions)

        preprocessing_time = time.time() - start_time

        return {
            "texts": texts,
            "languages": languages,
            "include_emotions_flags": include_emotions_flags,
            "preprocessing_time": preprocessing_time
        }

    def predict(self, input_list: List[TextInput], preprocessed_data: dict) -> dict:
        """Perform sentiment analysis on preprocessed texts"""
        import time
        start_time = time.time()

        texts = preprocessed_data["texts"]
        languages = preprocessed_data["languages"]
        include_emotions_flags = preprocessed_data["include_emotions_flags"]

        # Batch sentiment analysis
        sentiment_results = self.sentiment_pipeline(texts)

        # Process emotion detection for texts that need it
        emotion_results = []
        for i, (text, include_emotions) in enumerate(zip(texts, include_emotions_flags)):
            if include_emotions:
                emotions = self._get_emotions(text)
                emotion_results.append(emotions)
            else:
                emotion_results.append(None)

        inference_time = time.time() - start_time

        return {
            "sentiment_results": sentiment_results,
            "emotion_results": emotion_results,
            "inference_time": inference_time
        }

    def postprocess(self, input_list: List[TextInput], raw_output: dict) -> List[SentimentOutput]:
        """Format sentiment analysis outputs"""
        sentiment_results = raw_output["sentiment_results"]
        emotion_results = raw_output["emotion_results"]
        inference_time = raw_output["inference_time"]

        outputs = []

        for i, (input_item, sentiment_result, emotions) in enumerate(zip(input_list, sentiment_results, emotion_results)):
            # Normalize sentiment
            sentiment_label, confidence, normalized_score = self._normalize_sentiment_score(
                sentiment_result["label"], sentiment_result["score"]
            )

            # Detect language if needed
            language = input_item.language
            if language == "auto":
                language = self._detect_language(input_item.text)

            outputs.append(SentimentOutput(
                text=input_item.text,
                sentiment=sentiment_label,
                confidence=confidence,
                score=normalized_score,
                emotions=emotions,
                language=language,
                processing_time=inference_time / len(input_list)
            ))

        return outputs
```

## 2. Configuration

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "sentiment-service"

  # Authentication
  auth_config:
    ssm_key: "/sentiment-service/api-key"
    auth_header: "x-api-key"

  # Container configuration
  build_config:
    image: "python:3.11"
    tag: "latest"
    workdir: "/app"
    env:
      TRANSFORMERS_CACHE: "/tmp/transformers_cache"
      HF_HOME: "/tmp/huggingface"
    extra_run_commands:
      # Install PyTorch
      - "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
      # Install NLP libraries
      - "pip install transformers tokenizers"
      # Install language detection
      - "pip install langdetect"

  # Deployment configuration
  deployment_config:
    gpu: "T4"  # Cost-effective GPU for transformer models
    concurrency_limit: 10
    container_idle_timeout: 600  # 10 minutes
    retries: 3
    memory: 8192  # 8GB RAM

    # Mount pre-trained models from cloud storage
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "sentiment-models"
        secret: "aws-credentials"
        key_prefix: "production/"
        read_only: true

    # Cache for model downloads
    volumes:
      "/tmp/transformers_cache": "transformers-cache"
      "/tmp/huggingface": "huggingface-cache"
    volume_reload_interval_seconds: 3600

  # Batch processing configuration
  batch_config:
    max_batch_size: 16  # Process multiple texts efficiently
    wait_ms: 100

  # Queue configuration for async processing
  queue_config:
    backend: "taskiq"
    broker_url: "redis://redis:6379"

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    device: "cuda"
    max_length: 512
  model_entries:
    sentiment_model:
      sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
      emotion_model: "j-hartmann/emotion-english-distilroberta-base"
      supported_languages: ["en", "es", "fr", "de", "it"]
```

## 3. Modal App

Create `app.py`:

```python
import modal
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from sentiment_model import SentimentAnalysisInference, TextInput, SentimentOutput

# Initialize Modalkit
modal_config = ModalConfig()
app = modal.App(name=modal_config.app_name)

# Define Modal app class
@app.cls(**modal_config.get_app_cls_settings())
class SentimentApp(ModalService):
    inference_implementation = SentimentAnalysisInference
    model_name: str = modal.parameter(default="sentiment_model")
    modal_utils: ModalConfig = modal_config

# Create API endpoints
@app.function(**modal_config.get_handler_settings())
@modal.asgi_app(**modal_config.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=SentimentApp,
        input_model=TextInput,
        output_model=SentimentOutput
    )

# Health check endpoint
@app.function()
def health_check():
    return {"status": "healthy", "service": "sentiment-analysis"}

if __name__ == "__main__":
    # For local development
    with modal.enable_local_development():
        pass
```

## 4. Dependencies

Create `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
langdetect>=1.0.9
numpy>=1.21.0
```

## 5. Usage Examples

### Single Text Analysis

```python
import requests

headers = {"x-api-key": "your-api-key"}

# Analyze sentiment of a single text
response = requests.post(
    "https://your-org--sentiment-service.modal.run/predict_sync",
    json={
        "text": "I absolutely love this new product! It's amazing.",
        "language": "en",
        "include_emotions": True
    },
    headers=headers
)

result = response.json()
print(f"Text: {result['text']}")
print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
print(f"Score: {result['score']:.3f}")
if result['emotions']:
    print("Emotions:")
    for emotion, score in result['emotions'].items():
        print(f"  {emotion}: {score:.3f}")
```

### Batch Processing

```python
import requests

headers = {"x-api-key": "your-api-key"}

# Analyze multiple texts at once
texts = [
    {"text": "This is the best day ever!", "include_emotions": True},
    {"text": "I'm feeling really disappointed about this.", "include_emotions": True},
    {"text": "The weather is okay today.", "include_emotions": False},
    {"text": "I can't believe how terrible this service is!", "include_emotions": True}
]

response = requests.post(
    "https://your-org--sentiment-service.modal.run/predict_batch",
    json=texts,
    headers=headers
)

results = response.json()
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['sentiment']} (score: {result['score']:.2f})")
```

### Async Processing

```python
import requests
import time

headers = {"x-api-key": "your-api-key"}

# Submit for async processing
response = requests.post(
    "https://your-org--sentiment-service.modal.run/predict_async",
    json={
        "text": "This is a long text that might take some time to process...",
        "language": "en",
        "include_emotions": True
    },
    headers=headers
)

message_id = response.json()["message_id"]
print(f"Analysis submitted: {message_id}")

# In production, you would use webhooks instead of polling
# This is just for demonstration
time.sleep(5)  # Wait for processing
```

## 6. Production Features

### Error Handling

```python
import requests

headers = {"x-api-key": "your-api-key"}

# Test error handling
try:
    response = requests.post(
        "https://your-org--sentiment-service.modal.run/predict_sync",
        json={
            "text": "",  # Empty text
            "language": "en"
        },
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Handled empty text: {result['sentiment']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"Request failed: {e}")
```

### Monitoring and Metrics

```python
# In production, you can add monitoring
import requests
import time

headers = {"x-api-key": "your-api-key"}

# Measure response time
start_time = time.time()
response = requests.post(
    "https://your-org--sentiment-service.modal.run/predict_sync",
    json={
        "text": "This is a test for monitoring response times.",
        "language": "en"
    },
    headers=headers
)
end_time = time.time()

if response.status_code == 200:
    result = response.json()
    print(f"Response time: {end_time - start_time:.3f}s")
    print(f"Model processing time: {result['processing_time']:.3f}s")
```

## 7. Advanced Configuration

### Multi-Language Support

```yaml
# modalkit.yaml - Multi-language configuration
model_settings:
  model_entries:
    sentiment_model:
      sentiment_model: "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Multilingual model
      emotion_model: "j-hartmann/emotion-english-distilroberta-base"
      supported_languages: ["en", "es", "fr", "de", "it", "pt", "ja", "zh"]
```

### High-Performance Configuration

```yaml
# modalkit.yaml - High-performance setup
deployment_config:
  gpu: "A10G"  # More powerful GPU
  concurrency_limit: 20
  container_idle_timeout: 1800  # 30 minutes

batch_config:
  max_batch_size: 32  # Larger batches
  wait_ms: 200
```

### Cost-Optimized Configuration

```yaml
# modalkit.yaml - Cost-optimized setup
deployment_config:
  gpu: null  # Use CPU for cost savings
  cpu: 4.0
  memory: 8192
  concurrency_limit: 15
  container_idle_timeout: 300  # 5 minutes
```

## 8. Deployment

### Local Testing

```bash
# Start local server
modal serve app.py

# Test the service
curl -X POST http://localhost:8000/predict_sync \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"text": "I love this tutorial!", "include_emotions": true}'
```

### Production Deployment

```bash
# Deploy to Modal
modal deploy app.py

# Check deployment status
modal app list

# View logs
modal logs -f sentiment-service
```

## Key Features Demonstrated

1. **Practical Use Case**: Real-world sentiment analysis service
2. **Batch Processing**: Efficient processing of multiple texts
3. **Cloud Storage**: Model loading from S3/GCS
4. **Error Handling**: Graceful handling of edge cases
5. **Multi-modal Output**: Sentiment + emotion detection
6. **Language Detection**: Automatic language detection
7. **Production Ready**: Proper configuration for different environments
8. **Monitoring**: Built-in performance metrics

This example provides a solid foundation for building production ML services with Modalkit, demonstrating core concepts while solving a real business problem.
