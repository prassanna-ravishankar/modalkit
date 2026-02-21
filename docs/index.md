# Modalkit Documentation

<p align="center">
  <img src="./modalkit.png" width="400" height="400"/>
</p>

<p align="center">
  <b>A framework layer for deploying ML models on Modal</b>
</p>

## What is Modalkit?

Modalkit is a lightweight framework built on [Modal](https://modal.com) that provides ML-specific patterns for deploying machine learning models on Modal's serverless infrastructure.

## Why use Modalkit?

Modalkit adds the following on top of Modal's serverless compute:

- **ML-Specific Patterns**: Standardized inference pipeline (preprocess → predict → postprocess)
- **Configuration Management**: YAML-based config for Modal deployments
- **Built-in Auth**: Authentication setup for ML APIs
- **Type Safety**: Pydantic models for request/response validation
- **Queue Integration**: Async inference with SQS/Taskiq support

## How it works

Modalkit wraps your ML model in a Modal-compatible structure:

```python
# Your ML code
class MyModel(InferencePipeline):
    def predict(self, inputs):
        return model.generate(inputs)

# Modalkit handles the Modal integration
@app.cls(**modal_config.get_app_cls_settings())
class MyApp(ModalService):
    inference_implementation = MyModel

# Deploy with Modal CLI
# modal deploy app.py
```

Under the hood, Modalkit:
1. Configures Modal container specs from YAML
2. Sets up FastAPI endpoints
3. Handles authentication middleware
4. Manages batch processing
5. Integrates with Modal's volume and secret systems

## Prerequisites

- Python 3.9+
- [Modal account](https://modal.com) and CLI installed
- Basic familiarity with Modal concepts

## Quick Start

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Build your first ML deployment on Modal

    [:octicons-arrow-right-24: Start here](guide/getting-started.md)

-   :material-file-cog:{ .lg .middle } __Configuration__

    ---

    Learn how Modalkit configures Modal resources

    [:octicons-arrow-right-24: Config guide](guide/configuration.md)

-   :material-cloud-upload:{ .lg .middle } __Deployment__

    ---

    Deploy to Modal's infrastructure

    [:octicons-arrow-right-24: Deploy guide](guide/deployment.md)

-   :material-book-open-variant:{ .lg .middle } __Examples__

    ---

    Real-world ML deployment examples

    [:octicons-arrow-right-24: View examples](examples/basic.md)

</div>

## Core Concepts

### InferencePipeline
Your model inherits from `InferencePipeline` and implements three methods:
- `preprocess()`: Prepare raw inputs for your model
- `predict()`: Run inference
- `postprocess()`: Format outputs

### Modal Integration
Modalkit automatically configures:
- Container images and dependencies
- GPU resources
- Secrets and volumes
- Concurrency limits
- CloudBucketMounts for S3/GCS access

### Configuration
A single YAML file configures your entire deployment:
```yaml
app_settings:
  deployment_config:
    gpu: "T4"
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "my-models"
```

## When to use Modalkit

**Use Modalkit for:**
- ML model deployment on Modal
- Standardized API patterns
- Configuration-driven deployments
- Built-in auth and validation

**Use Modal directly for:**
- Non-ML workloads
- Custom networking requirements
- Fine-grained container control

## Learn More

- [Modal Documentation](https://modal.com/docs) - Understand the platform
- [Modalkit GitHub](https://github.com/prassanna-ravishankar/modalkit) - Source code and issues
- [Examples](examples/basic.md) - Working code examples

---

<p align="center">
  Modalkit is an open-source project • Not affiliated with Modal Labs
</p>
