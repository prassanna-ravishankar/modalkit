# Configuration Guide

Modalkit uses YAML configuration to manage Modal deployments. This guide covers all configuration options.

## Configuration File

By default, Modalkit looks for `modalkit.yaml` in your project root. You can override this with the `MODALKIT_CONFIG` environment variable:

```bash
export MODALKIT_CONFIG="config/production.yaml"

# Or use multiple configs (later ones override earlier)
export MODALKIT_CONFIG="base.yaml,production.yaml"
```

## Configuration Structure

```yaml
app_settings:
  app_prefix: string          # Prefix for Modal app name
  auth_config: AuthConfig     # Authentication settings
  build_config: BuildConfig   # Container configuration
  deployment_config: DeploymentConfig  # Runtime settings
  batch_config: BatchConfig   # Batch processing settings
  queue_config: QueueConfig   # Async queue settings

model_settings:
  local_model_repository_folder: string  # Path to models
  common: dict              # Shared settings
  model_entries: dict       # Per-model configurations
```

## App Settings

### app_prefix
The prefix for your Modal app name. The full name will be: `{app_prefix}{MODALKIT_APP_POSTFIX}`

```yaml
app_settings:
  app_prefix: "my-service"  # Results in "my-service-dev" by default
```

### auth_config
Configure authentication for your endpoints.

#### Option 1: API Key Authentication
```yaml
auth_config:
  # Use AWS SSM (recommended for production)
  ssm_key: "/myapp/api-key"
  auth_header: "x-api-key"

  # OR hardcode (dev only)
  # api_key: "sk-12345"
  # auth_header: "x-api-key"
```

#### Option 2: Modal Proxy Auth
```yaml
deployment_config:
  secure: true  # Enables Modal proxy authentication
```

### build_config
Configure the Modal container image.

```yaml
build_config:
  # Base image - can be any Docker image
  image: "python:3.11-slim"
  tag: "latest"

  # Working directory in container
  workdir: "/app"

  # Environment variables
  env:
    MODEL_VERSION: "1.0"
    DEBUG: "false"

  # Additional commands to run during build
  extra_run_commands:
    - "apt-get update && apt-get install -y libgl1"
    - "pip install torch torchvision"
```

### deployment_config
Runtime configuration for your Modal deployment.

```yaml
deployment_config:
  # GPU configuration
  gpu: "T4"  # Options: T4, A10G, A100, H100, null (CPU only)

  # Concurrency settings
  concurrency_limit: 10  # Max concurrent containers
  allow_concurrent_inputs: 5  # Inputs per container
  container_idle_timeout: 300  # Seconds before shutdown

  # CPU/Memory (when gpu is null)
  memory: 8192  # MB
  cpu: 4.0      # CPU cores

  # Retry configuration
  retries: 3

  # Modal region
  region: "us-east-1"  # or "us-west-2"

  # Authentication mode
  secure: false  # true for Modal proxy auth

  # Modal volumes
  volumes:
    "/mnt/cache": "model-cache-vol"
  volume_reload_interval_seconds: 300  # Auto-reload interval

  # File mounts
  mounts:
    - local_path: "configs/prod.json"
      remote_path: "/app/config.json"
      type: "file"
    - local_path: "data/"
      remote_path: "/app/data/"
      type: "dir"

  # Cloud storage mounts
  cloud_bucket_mounts:
    - mount_point: "/mnt/models"
      bucket_name: "my-models"
      secret: "aws-creds"
      key_prefix: "v1/"
      read_only: true
```

### batch_config
Configure batch processing behavior.

```yaml
batch_config:
  max_batch_size: 32  # Maximum items per batch
  wait_ms: 100        # Max wait time to fill batch
```

### queue_config
For async endpoints, configure the queue backend.

```yaml
queue_config:
  # For AWS SQS
  backend: "sqs"

  # For Taskiq
  backend: "taskiq"
  broker_url: "redis://localhost:6379"  # or "memory://" for dev
```

## Model Settings

Configure model-specific settings.

```yaml
model_settings:
  # Base directory for model files
  local_model_repository_folder: "./models"

  # Settings shared across all models
  common:
    device: "cuda"  # or "cpu"
    cache_dir: "/tmp/cache"
    max_length: 512

  # Per-model configurations
  model_entries:
    gpt2_model:
      path: "models/gpt2"
      temperature: 0.7
      top_p: 0.9

    bert_model:
      path: "models/bert"
      max_seq_length: 128
```

## Cloud Storage Configuration

### AWS S3
```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/s3-models"
    bucket_name: "my-ml-models"
    secret: "aws-credentials"  # Modal secret name
    key_prefix: "production/"
    read_only: true
```

Create the secret:
```bash
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=xxx \
  AWS_SECRET_ACCESS_KEY=yyy \
  AWS_DEFAULT_REGION=us-east-1
```

### Google Cloud Storage
```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/gcs-data"
    bucket_name: "my-data"
    bucket_endpoint_url: "https://storage.googleapis.com"
    secret: "gcp-credentials"
```

### Cloudflare R2
```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/r2"
    bucket_name: "my-bucket"
    bucket_endpoint_url: "https://accountid.r2.cloudflarestorage.com"
    secret: "r2-credentials"
```

## Environment Variables

Override configuration with environment variables:

```bash
# Override app prefix
export MODALKIT_APP_SETTINGS__APP_PREFIX="prod-service"

# Override GPU
export MODALKIT_APP_SETTINGS__DEPLOYMENT_CONFIG__GPU="A100"

# Set config file
export MODALKIT_CONFIG="production.yaml"

# Set app postfix
export MODALKIT_APP_POSTFIX="-prod"  # Results in "my-service-prod"
```

## Configuration Examples

### Development Config
```yaml
app_settings:
  app_prefix: "ml-dev"
  auth_config:
    api_key: "dev-key"
    auth_header: "x-api-key"
  build_config:
    image: "python:3.11"
    tag: "latest"
  deployment_config:
    gpu: null
    concurrency_limit: 1
    secure: false
  batch_config:
    max_batch_size: 1
    wait_ms: 0
```

### Production Config
```yaml
app_settings:
  app_prefix: "ml-prod"
  auth_config:
    ssm_key: "/prod/api-key"
    auth_header: "x-api-key"
  build_config:
    image: "ghcr.io/myorg/ml-base:stable"
    tag: "v1.2.3"
  deployment_config:
    gpu: "A100"
    concurrency_limit: 50
    secure: true  # Use Modal proxy auth
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "prod-models"
        secret: "aws-prod"
        read_only: true
  batch_config:
    max_batch_size: 64
    wait_ms: 200
```

## Best Practices

1. **Use environment-specific configs**: Separate dev/staging/prod configurations
2. **Store secrets in Modal**: Use `modal secret create` for sensitive data
3. **Version your images**: Use specific tags, not `latest` in production
4. **Set appropriate timeouts**: Balance cost vs responsiveness
5. **Monitor concurrency**: Start conservative, increase based on load
6. **Use read-only mounts**: For model artifacts and reference data
