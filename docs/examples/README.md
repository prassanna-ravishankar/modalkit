# Modalkit Examples

Comprehensive examples showcasing Modalkit's capabilities for deploying production-ready ML services on Modal.

## 🎯 Getting Started

### [Sentiment Analysis Service](basic.md)
A complete sentiment analysis API demonstrating core Modalkit features.

**What you'll learn:**
- Basic Modalkit setup and configuration
- Text processing with transformers
- Batch processing and error handling
- Cloud storage integration
- Production deployment patterns

**Technologies:** Transformers, PyTorch, Language Detection
**Difficulty:** ⭐⭐☆☆☆

---

## 🚀 Advanced Examples

### [Async Processing with Queue Backends](async.md)
Flexible async processing with TaskIQ, SQS, or custom queue systems.

**What you'll learn:**
- Optional queue backend integration
- TaskIQ dependency injection
- Custom queue implementations
- Error-resilient async processing
- Production queue setup patterns

**Technologies:** TaskIQ, Redis, SQS, RabbitMQ
**Difficulty:** ⭐⭐⭐☆☆

**🔗 Working Examples:** See [queue-patterns.md](queue-patterns.md) and [taskiq-integration.md](taskiq-integration.md)

---

### [Large Language Model Deployment](llm-deployment.md)
Deploy and scale LLMs with GPU acceleration and intelligent batching.

**What you'll learn:**
- GPU optimization for LLMs
- Model loading from cloud storage
- Intelligent batching strategies
- Memory management techniques
- Cost optimization approaches

**Technologies:** Transformers, PyTorch, HuggingFace Hub
**Difficulty:** ⭐⭐⭐⭐☆

---

### [Computer Vision Pipeline](computer-vision.md)
Production computer vision service with classification and object detection.

**What you'll learn:**
- Multi-task computer vision
- Image preprocessing and augmentation
- GPU acceleration for vision models
- Object detection with YOLO
- Performance monitoring

**Technologies:** OpenCV, Torchvision, Ultralytics, YOLO
**Difficulty:** ⭐⭐⭐☆☆

---

### [Multi-Modal AI Service](multimodal-ai.md)
Sophisticated AI service processing text, images, and audio simultaneously.

**What you'll learn:**
- Cross-modal understanding
- Vector embeddings generation
- Semantic search capabilities
- Advanced model orchestration
- Complex input validation

**Technologies:** CLIP, Whisper, Sentence Transformers, Vector Databases
**Difficulty:** ⭐⭐⭐⭐⭐

---

### [Real-Time Analytics Pipeline](realtime-analytics.md)
Stream processing and ML analytics for time-series data and events.

**What you'll learn:**
- Stream processing patterns
- Anomaly detection algorithms
- Time-series forecasting
- Event classification and alerting
- Redis integration for real-time data

**Technologies:** Redis, Scikit-learn, Prophet, Pandas
**Difficulty:** ⭐⭐⭐⭐☆

---

## 📊 Example Comparison

| Example | Use Case | Complexity | GPU Required | Key Features |
|---------|----------|------------|--------------|--------------|
| [Sentiment Analysis](basic.md) | Text analysis API | Basic | Optional | Batch processing, cloud storage |
| [Async Processing](async.md) | Queue-based workflows | Intermediate | Optional | TaskIQ integration, dependency injection |
| [LLM Deployment](llm-deployment.md) | Text generation | Advanced | Yes | GPU optimization, large models |
| [Computer Vision](computer-vision.md) | Image analysis | Intermediate | Recommended | Multi-task, object detection |
| [Multi-Modal AI](multimodal-ai.md) | Cross-modal understanding | Expert | Yes | Complex orchestration, embeddings |
| [Real-Time Analytics](realtime-analytics.md) | Stream processing | Advanced | No | Time-series, real-time processing |

## 🛠️ Development Patterns

Each example demonstrates different Modalkit patterns:

### **Basic Patterns** (Sentiment Analysis)
- Simple inference pipeline
- Configuration-driven deployment
- Basic error handling
- Standard authentication

### **Async Patterns** (Async Processing)
- Optional queue backend integration
- TaskIQ dependency injection
- Custom queue implementations
- Error-resilient async processing

### **Performance Patterns** (LLM, Computer Vision)
- GPU acceleration techniques
- Memory optimization
- Intelligent batching
- Model caching strategies

### **Advanced Patterns** (Multi-Modal, Analytics)
- Complex data orchestration
- Multiple model coordination
- Real-time processing
- Advanced monitoring

### **Production Patterns** (All Examples)
- Cloud storage integration
- Comprehensive error handling
- Monitoring and observability
- Auto-scaling configuration

## 📁 Example Structure

Each example follows a consistent structure:

```
example-name/
├── README.md              # Tutorial walkthrough
├── app.py                 # Modal app definition
├── model.py               # ML model implementation
├── modalkit.yaml          # Configuration
├── requirements.txt       # Dependencies
├── utils/                 # Utility functions
│   ├── preprocessing.py
│   └── validation.py
└── tests/                 # Test files
    ├── test_model.py
    └── test_api.py
```

## 🚀 Quick Start

1. **Choose an example** based on your use case
2. **Follow the tutorial** step-by-step
3. **Customize** for your specific needs
4. **Deploy** to production

### Prerequisites

- [Modal account](https://modal.com) and CLI installed
- Python 3.9+
- Basic familiarity with ML concepts

### Installation

```bash
# Install Modalkit
pip install modalkit

# Install Modal CLI
pip install modal
modal setup
```

## 🎯 Use Case Guide

### **Text Processing**
- **Simple text analysis** → [Sentiment Analysis](basic.md)
- **Text generation** → [LLM Deployment](llm-deployment.md)
- **Multi-language support** → [Multi-Modal AI](multimodal-ai.md)

### **Async Processing**
- **Queue-based workflows** → [Async Processing](async.md)
- **TaskIQ integration** → [Async Processing](async.md)
- **Custom queue systems** → [Async Processing](async.md)

### **Computer Vision**
- **Image classification** → [Computer Vision](computer-vision.md)
- **Object detection** → [Computer Vision](computer-vision.md)
- **Visual search** → [Multi-Modal AI](multimodal-ai.md)

### **Audio Processing**
- **Speech recognition** → [Multi-Modal AI](multimodal-ai.md)
- **Audio classification** → [Multi-Modal AI](multimodal-ai.md)

### **Analytics & Monitoring**
- **Real-time metrics** → [Real-Time Analytics](realtime-analytics.md)
- **Anomaly detection** → [Real-Time Analytics](realtime-analytics.md)
- **Forecasting** → [Real-Time Analytics](realtime-analytics.md)

## 📊 Performance Benchmarks

### **Latency Expectations**
- **Sentiment Analysis:** ~50-100ms per request
- **LLM Generation:** ~1-5s per request (depends on length)
- **Computer Vision:** ~100-300ms per image
- **Multi-Modal:** ~200-500ms per request
- **Real-Time Analytics:** ~10-50ms per event

### **Throughput Expectations**
- **Text Processing:** 100-1000 requests/second
- **Image Processing:** 10-100 images/second
- **Audio Processing:** 5-50 files/second
- **Analytics:** 1000-10000 events/second

## 🔧 Configuration Templates

### **Development Environment**
```yaml
deployment_config:
  gpu: null
  concurrency_limit: 1
  container_idle_timeout: 300
  secure: false
```

### **Production Environment**
```yaml
deployment_config:
  gpu: "T4"  # or A10G/A100 for heavier workloads
  concurrency_limit: 10
  container_idle_timeout: 900
  secure: true
  retries: 3
```

### **High-Scale Environment**
```yaml
deployment_config:
  gpu: "A100"
  concurrency_limit: 50
  container_idle_timeout: 1800
  memory: 32768
  retries: 5
```

## 🤝 Contributing

Want to add your own example? See our [Contributing Guidelines](https://github.com/prassanna-ravishankar/modalkit/blob/main/CONTRIBUTING.md) for:

- Example requirements and standards
- Code quality expectations
- Documentation guidelines
- Testing requirements

## 📚 Additional Resources

- [Modalkit Documentation](../index.md)
- [Configuration Guide](../guide/configuration.md)
- [Deployment Guide](../guide/deployment.md)
- [Modal Documentation](https://modal.com/docs)

## 💡 Need Help?

- **Issues:** [GitHub Issues](https://github.com/prassanna-ravishankar/modalkit/issues)
- **Discussions:** [GitHub Discussions](https://github.com/prassanna-ravishankar/modalkit/discussions)
- **Examples:** Each example includes troubleshooting sections

---

These examples demonstrate Modalkit's power in deploying production-ready ML services with minimal boilerplate code. Start with the sentiment analysis example and progress to more complex use cases as you become familiar with the framework.
