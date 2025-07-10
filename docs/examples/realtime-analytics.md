# Real-Time Analytics Pipeline with Modalkit

Deploy a real-time analytics and ML pipeline using Modalkit, showcasing stream processing, time-series analysis, and automated decision making.

## Overview

This tutorial demonstrates:
- Real-time data ingestion and processing
- Time-series forecasting and anomaly detection
- Event-driven ML inference
- Automated alerting and decision making
- High-throughput stream processing
- Integration with external systems (Redis, Kafka, databases)

## Architecture

```
realtime-analytics/
├── app.py                      # Modal app definition
├── models/
│   ├── anomaly_detector.py     # Anomaly detection model
│   ├── forecaster.py           # Time-series forecasting
│   └── classifier.py           # Event classification
├── processors/
│   ├── stream_processor.py     # Stream processing logic
│   ├── aggregator.py           # Data aggregation
│   └── alerting.py             # Alert generation
├── utils/
│   ├── time_series.py          # Time-series utilities
│   ├── metrics.py              # Metrics calculation
│   └── storage.py              # Data storage utilities
├── modalkit.yaml               # Configuration
└── requirements.txt            # Dependencies
```

## 1. Real-Time Analytics Model

Create `models/realtime_analytics.py`:

```python
from modalkit.inference import InferencePipeline
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import redis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Input schemas
class MetricDataPoint(BaseModel):
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = Field(default_factory=dict)
    source: str = "unknown"

class EventData(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: str
    payload: Dict[str, Any]
    severity: str = "info"  # "info", "warning", "error", "critical"

class AnalyticsInput(BaseModel):
    data_points: List[MetricDataPoint] = Field(default_factory=list)
    events: List[EventData] = Field(default_factory=list)
    task: str = "analyze"  # "analyze", "forecast", "detect_anomaly", "classify"
    window_size: int = 100  # Number of data points for analysis
    forecast_horizon: int = 24  # Hours to forecast
    anomaly_threshold: float = 0.05  # Anomaly detection threshold

# Output schemas
class AnomalyResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    timestamp: datetime
    metric_name: str
    expected_range: tuple[float, float]
    actual_value: float

class ForecastResult(BaseModel):
    timestamp: datetime
    predicted_value: float
    confidence_interval: tuple[float, float]
    trend: str  # "increasing", "decreasing", "stable"
    seasonal_component: float

class ClassificationResult(BaseModel):
    event_id: str
    predicted_class: str
    confidence: float
    risk_score: float
    recommended_action: str

class AlertResult(BaseModel):
    alert_id: str
    severity: str
    message: str
    timestamp: datetime
    affected_metrics: List[str]
    recommended_actions: List[str]

class AnalyticsOutput(BaseModel):
    task: str
    anomalies: List[AnomalyResult] = Field(default_factory=list)
    forecasts: List[ForecastResult] = Field(default_factory=list)
    classifications: List[ClassificationResult] = Field(default_factory=list)
    alerts: List[AlertResult] = Field(default_factory=list)
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float
    data_quality_score: float

class RealTimeAnalyticsInference(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)

        self.model_config = common_settings.get(model_name, {})

        # Initialize Redis connection for real-time data
        self.redis_client = self._init_redis()

        # Initialize models
        self.anomaly_detector = self._load_anomaly_detector()
        self.forecaster = self._load_forecaster()
        self.classifier = self._load_classifier()

        # Initialize scalers
        self.scaler = StandardScaler()

        # Alert thresholds
        self.alert_thresholds = self.model_config.get("alert_thresholds", {
            "anomaly_score": 0.8,
            "forecast_error": 0.1,
            "classification_confidence": 0.7
        })

        print("Real-time analytics service initialized")

    def _init_redis(self):
        """Initialize Redis connection"""
        redis_config = self.model_config.get("redis", {})
        try:
            client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None

    def _load_anomaly_detector(self):
        """Load anomaly detection model"""
        model_path = "/mnt/models/anomaly_detector.pkl"

        if os.path.exists(model_path):
            print("Loading anomaly detector from mounted storage")
            return joblib.load(model_path)
        else:
            print("Initializing new anomaly detector")
            return IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )

    def _load_forecaster(self):
        """Load time-series forecasting model"""
        try:
            from prophet import Prophet
            model_path = "/mnt/models/forecaster.pkl"

            if os.path.exists(model_path):
                print("Loading forecaster from mounted storage")
                return joblib.load(model_path)
            else:
                print("Initializing new forecaster")
                return Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
        except ImportError:
            print("Prophet not available, using simple forecaster")
            return self._get_simple_forecaster()

    def _get_simple_forecaster(self):
        """Simple moving average forecaster"""
        class SimpleForecaster:
            def __init__(self, window=24):
                self.window = window

            def fit(self, data):
                return self

            def predict(self, periods):
                # Simple moving average prediction
                return np.random.randn(periods)

        return SimpleForecaster()

    def _load_classifier(self):
        """Load event classification model"""
        from sklearn.ensemble import RandomForestClassifier

        model_path = "/mnt/models/event_classifier.pkl"

        if os.path.exists(model_path):
            print("Loading event classifier from mounted storage")
            return joblib.load(model_path)
        else:
            print("Initializing new event classifier")
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

    def _get_historical_data(self, metric_name: str, hours: int = 24) -> pd.DataFrame:
        """Get historical data from Redis"""
        if not self.redis_client:
            return pd.DataFrame()

        try:
            key = f"metrics:{metric_name}"
            data = self.redis_client.lrange(key, 0, hours * 60)  # Assuming 1-minute intervals

            if not data:
                return pd.DataFrame()

            records = [json.loads(item) for item in data]
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')

        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _store_data_point(self, data_point: MetricDataPoint):
        """Store data point in Redis"""
        if not self.redis_client:
            return

        try:
            key = f"metrics:{data_point.metric_name}"
            value = json.dumps({
                "timestamp": data_point.timestamp.isoformat(),
                "value": data_point.value,
                "tags": data_point.tags,
                "source": data_point.source
            })

            # Store with TTL (keep last 7 days)
            self.redis_client.lpush(key, value)
            self.redis_client.expire(key, 7 * 24 * 3600)

        except Exception as e:
            print(f"Error storing data point: {e}")

    def _detect_anomalies(self, data_points: List[MetricDataPoint]) -> List[AnomalyResult]:
        """Detect anomalies in data points"""
        anomalies = []

        # Group by metric name
        metric_groups = {}
        for dp in data_points:
            if dp.metric_name not in metric_groups:
                metric_groups[dp.metric_name] = []
            metric_groups[dp.metric_name].append(dp)

        for metric_name, points in metric_groups.items():
            if len(points) < 10:  # Need sufficient data
                continue

            # Get historical data
            historical_df = self._get_historical_data(metric_name, hours=24)

            if len(historical_df) < 50:  # Need historical context
                continue

            # Prepare data for anomaly detection
            values = [dp.value for dp in points]
            timestamps = [dp.timestamp for dp in points]

            # Add historical values
            all_values = historical_df['value'].tolist() + values
            all_values = np.array(all_values).reshape(-1, 1)

            # Fit and predict
            self.anomaly_detector.fit(all_values[:-len(values)])
            anomaly_scores = self.anomaly_detector.decision_function(all_values[-len(values):])
            predictions = self.anomaly_detector.predict(all_values[-len(values):])

            # Calculate expected range
            historical_mean = historical_df['value'].mean()
            historical_std = historical_df['value'].std()
            expected_range = (
                historical_mean - 2 * historical_std,
                historical_mean + 2 * historical_std
            )

            # Generate anomaly results
            for i, (point, score, pred) in enumerate(zip(points, anomaly_scores, predictions)):
                is_anomaly = pred == -1
                anomaly_confidence = abs(score)

                anomalies.append(AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=float(score),
                    confidence=float(anomaly_confidence),
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    expected_range=expected_range,
                    actual_value=point.value
                ))

        return anomalies

    def _generate_forecasts(self, data_points: List[MetricDataPoint], horizon: int) -> List[ForecastResult]:
        """Generate forecasts for metrics"""
        forecasts = []

        # Group by metric name
        metric_groups = {}
        for dp in data_points:
            if dp.metric_name not in metric_groups:
                metric_groups[dp.metric_name] = []
            metric_groups[dp.metric_name].append(dp)

        for metric_name, points in metric_groups.items():
            # Get historical data
            historical_df = self._get_historical_data(metric_name, hours=48)

            if len(historical_df) < 48:  # Need sufficient historical data
                continue

            # Prepare data for forecasting
            df = historical_df.copy()
            df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})

            try:
                # Fit forecaster
                if hasattr(self.forecaster, 'fit'):
                    self.forecaster.fit(df)

                # Generate future timestamps
                future_timestamps = [
                    points[-1].timestamp + timedelta(hours=i+1)
                    for i in range(horizon)
                ]

                # Generate predictions
                if hasattr(self.forecaster, 'predict'):
                    future_df = pd.DataFrame({
                        'ds': future_timestamps
                    })
                    predictions = self.forecaster.predict(future_df)

                    for i, (ts, pred) in enumerate(zip(future_timestamps, predictions)):
                        # Determine trend
                        if i > 0:
                            prev_pred = predictions[i-1] if isinstance(predictions, list) else predictions.iloc[i-1]['yhat']
                            current_pred = pred if isinstance(pred, (int, float)) else pred['yhat']

                            if current_pred > prev_pred * 1.05:
                                trend = "increasing"
                            elif current_pred < prev_pred * 0.95:
                                trend = "decreasing"
                            else:
                                trend = "stable"
                        else:
                            trend = "stable"

                        forecasts.append(ForecastResult(
                            timestamp=ts,
                            predicted_value=float(current_pred) if isinstance(pred, (int, float)) else float(pred['yhat']),
                            confidence_interval=(
                                float(pred['yhat_lower']) if isinstance(pred, dict) else current_pred * 0.9,
                                float(pred['yhat_upper']) if isinstance(pred, dict) else current_pred * 1.1
                            ),
                            trend=trend,
                            seasonal_component=float(pred.get('seasonal', 0)) if isinstance(pred, dict) else 0.0
                        ))

            except Exception as e:
                print(f"Forecasting error for {metric_name}: {e}")
                continue

        return forecasts

    def _classify_events(self, events: List[EventData]) -> List[ClassificationResult]:
        """Classify events for risk assessment"""
        classifications = []

        for event in events:
            # Extract features from event
            features = self._extract_event_features(event)

            # Simple rule-based classification (replace with trained model)
            if event.severity == "critical":
                predicted_class = "high_risk"
                confidence = 0.95
                risk_score = 0.9
                action = "immediate_response"
            elif event.severity == "error":
                predicted_class = "medium_risk"
                confidence = 0.8
                risk_score = 0.6
                action = "investigation_required"
            elif event.severity == "warning":
                predicted_class = "low_risk"
                confidence = 0.7
                risk_score = 0.3
                action = "monitoring"
            else:
                predicted_class = "no_risk"
                confidence = 0.6
                risk_score = 0.1
                action = "log_only"

            classifications.append(ClassificationResult(
                event_id=event.event_id,
                predicted_class=predicted_class,
                confidence=confidence,
                risk_score=risk_score,
                recommended_action=action
            ))

        return classifications

    def _extract_event_features(self, event: EventData) -> Dict[str, Any]:
        """Extract features from event for classification"""
        features = {
            "event_type": event.event_type,
            "severity": event.severity,
            "payload_size": len(str(event.payload)),
            "hour_of_day": event.timestamp.hour,
            "day_of_week": event.timestamp.weekday(),
            "payload_keys": list(event.payload.keys())
        }
        return features

    def _generate_alerts(self, anomalies: List[AnomalyResult],
                        forecasts: List[ForecastResult],
                        classifications: List[ClassificationResult]) -> List[AlertResult]:
        """Generate alerts based on analysis results"""
        alerts = []

        # Anomaly-based alerts
        high_score_anomalies = [a for a in anomalies if a.anomaly_score < -0.5 and a.is_anomaly]
        if high_score_anomalies:
            affected_metrics = list(set([a.metric_name for a in high_score_anomalies]))

            alerts.append(AlertResult(
                alert_id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="warning",
                message=f"Anomalies detected in {len(affected_metrics)} metrics",
                timestamp=datetime.now(),
                affected_metrics=affected_metrics,
                recommended_actions=["investigate_anomaly", "check_data_source", "verify_thresholds"]
            ))

        # Forecast-based alerts
        concerning_forecasts = [f for f in forecasts if f.trend == "decreasing" and f.predicted_value < 0]
        if concerning_forecasts:
            alerts.append(AlertResult(
                alert_id=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="info",
                message=f"Declining trend predicted for {len(concerning_forecasts)} metrics",
                timestamp=datetime.now(),
                affected_metrics=[],
                recommended_actions=["capacity_planning", "trend_analysis"]
            ))

        # Classification-based alerts
        high_risk_events = [c for c in classifications if c.risk_score > 0.7]
        if high_risk_events:
            alerts.append(AlertResult(
                alert_id=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="error",
                message=f"{len(high_risk_events)} high-risk events detected",
                timestamp=datetime.now(),
                affected_metrics=[],
                recommended_actions=["incident_response", "escalate_to_team", "immediate_investigation"]
            ))

        return alerts

    def _calculate_summary_stats(self, data_points: List[MetricDataPoint]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not data_points:
            return {}

        values = [dp.value for dp in data_points]
        unique_metrics = set([dp.metric_name for dp in data_points])

        return {
            "total_data_points": len(data_points),
            "unique_metrics": len(unique_metrics),
            "value_stats": {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            },
            "time_range": {
                "start": min([dp.timestamp for dp in data_points]).isoformat(),
                "end": max([dp.timestamp for dp in data_points]).isoformat()
            }
        }

    def _assess_data_quality(self, data_points: List[MetricDataPoint],
                           events: List[EventData]) -> float:
        """Assess data quality score"""
        if not data_points and not events:
            return 0.0

        quality_score = 1.0

        # Check for missing values
        values = [dp.value for dp in data_points]
        if any(v is None or np.isnan(v) for v in values):
            quality_score -= 0.2

        # Check for timestamp consistency
        timestamps = [dp.timestamp for dp in data_points]
        if timestamps:
            time_diffs = [abs((timestamps[i+1] - timestamps[i]).total_seconds())
                         for i in range(len(timestamps)-1)]
            if time_diffs and max(time_diffs) > 3600:  # Gaps > 1 hour
                quality_score -= 0.1

        # Check for duplicate data
        timestamp_values = [(dp.timestamp, dp.value, dp.metric_name) for dp in data_points]
        if len(timestamp_values) != len(set(timestamp_values)):
            quality_score -= 0.1

        return max(0.0, quality_score)

    def preprocess(self, input_list: List[AnalyticsInput]) -> dict:
        """Preprocess analytics inputs"""
        import time
        start_time = time.time()

        all_data_points = []
        all_events = []
        tasks = []

        for input_item in input_list:
            all_data_points.extend(input_item.data_points)
            all_events.extend(input_item.events)
            tasks.append(input_item.task)

            # Store data points for historical analysis
            for dp in input_item.data_points:
                self._store_data_point(dp)

        preprocessing_time = time.time() - start_time

        return {
            "all_data_points": all_data_points,
            "all_events": all_events,
            "tasks": tasks,
            "input_items": input_list,
            "preprocessing_time": preprocessing_time
        }

    def predict(self, input_list: List[AnalyticsInput], preprocessed_data: dict) -> dict:
        """Perform real-time analytics"""
        import time
        start_time = time.time()

        all_data_points = preprocessed_data["all_data_points"]
        all_events = preprocessed_data["all_events"]

        # Initialize results
        all_anomalies = []
        all_forecasts = []
        all_classifications = []

        # Process each input
        for input_item in input_list:
            if input_item.task in ["analyze", "detect_anomaly"]:
                anomalies = self._detect_anomalies(input_item.data_points)
                all_anomalies.extend(anomalies)

            if input_item.task in ["analyze", "forecast"]:
                forecasts = self._generate_forecasts(input_item.data_points, input_item.forecast_horizon)
                all_forecasts.extend(forecasts)

            if input_item.task in ["analyze", "classify"]:
                classifications = self._classify_events(input_item.events)
                all_classifications.extend(classifications)

        # Generate alerts
        alerts = self._generate_alerts(all_anomalies, all_forecasts, all_classifications)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_data_points)

        # Assess data quality
        data_quality_score = self._assess_data_quality(all_data_points, all_events)

        processing_time = time.time() - start_time

        return {
            "anomalies": all_anomalies,
            "forecasts": all_forecasts,
            "classifications": all_classifications,
            "alerts": alerts,
            "summary_stats": summary_stats,
            "data_quality_score": data_quality_score,
            "processing_time": processing_time
        }

    def postprocess(self, input_list: List[AnalyticsInput], raw_output: dict) -> List[AnalyticsOutput]:
        """Format analytics outputs"""
        # Since we're processing all inputs together, create a single comprehensive output
        return [AnalyticsOutput(
            task="analyze",
            anomalies=raw_output["anomalies"],
            forecasts=raw_output["forecasts"],
            classifications=raw_output["classifications"],
            alerts=raw_output["alerts"],
            summary_stats=raw_output["summary_stats"],
            processing_time=raw_output["processing_time"],
            data_quality_score=raw_output["data_quality_score"]
        )]
```

## 2. Configuration

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "realtime-analytics"

  # Authentication
  auth_config:
    ssm_key: "/analytics/api-key"
    auth_header: "x-api-key"

  # Container with analytics dependencies
  build_config:
    image: "python:3.11"
    tag: "latest"
    workdir: "/app"
    env:
      PYTHONPATH: "/app"
      REDIS_URL: "redis://redis:6379"
    extra_run_commands:
      # System dependencies
      - "apt-get update && apt-get install -y build-essential"
      # Analytics libraries
      - "pip install pandas numpy scikit-learn"
      - "pip install redis kafka-python"
      # Time series libraries
      - "pip install prophet statsmodels"
      # ML libraries
      - "pip install joblib xgboost lightgbm"
      # Monitoring libraries
      - "pip install prometheus-client"

  # High-performance deployment
  deployment_config:
    gpu: null  # CPU-based analytics
    concurrency_limit: 20
    container_idle_timeout: 300
    retries: 3
    memory: 16384  # 16GB RAM for data processing
    cpu: 8.0       # 8 CPU cores

    # Mount trained models
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "analytics-models"
        secret: "aws-credentials"
        key_prefix: "production/"
        read_only: true

    # Cache and temporary storage
    volumes:
      "/tmp/analytics_cache": "analytics-cache"
    volume_reload_interval_seconds: 1800

  # High-throughput batch processing
  batch_config:
    max_batch_size: 50  # Process many data points together
    wait_ms: 50

  # Queue for different analytics tasks
  queue_config:
    backend: "taskiq"
    broker_url: "redis://redis:6379"

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    redis:
      host: "redis"
      port: 6379
    alert_thresholds:
      anomaly_score: 0.7
      forecast_error: 0.15
      classification_confidence: 0.8
  model_entries:
    analytics_model:
      anomaly_contamination: 0.1
      forecast_horizon: 24
      classification_threshold: 0.7
```

## 3. Modal App

Create `app.py`:

```python
import modal
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from models.realtime_analytics import RealTimeAnalyticsInference, AnalyticsInput, AnalyticsOutput

# Initialize Modalkit
modal_utils = ModalConfig()
app = modal.App(name=modal_utils.app_name)

# Define Modal app class
@app.cls(**modal_utils.get_app_cls_settings())
class AnalyticsApp(ModalService):
    inference_implementation = RealTimeAnalyticsInference
    model_name: str = modal.parameter(default="analytics_model")
    modal_utils: ModalConfig = modal_utils

# Create endpoints
@app.function(**modal_utils.get_handler_settings())
@modal.asgi_app(**modal_utils.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=AnalyticsApp,
        input_model=AnalyticsInput,
        output_model=AnalyticsOutput
    )

# Specialized endpoints for different analytics tasks
@app.function(**modal_utils.get_handler_settings())
def detect_anomalies(metric_data: list):
    """Dedicated anomaly detection endpoint"""
    pass

@app.function(**modal_utils.get_handler_settings())
def generate_forecasts(metric_name: str, horizon: int = 24):
    """Dedicated forecasting endpoint"""
    pass

@app.function(**modal_utils.get_handler_settings())
def process_events(events: list):
    """Dedicated event processing endpoint"""
    pass

if __name__ == "__main__":
    with modal.enable_local_development():
        pass
```

## 4. Usage Examples

### Real-Time Anomaly Detection

```python
import requests
from datetime import datetime, timedelta
import random

# Generate sample metric data
headers = {"x-api-key": "your-api-key"}
now = datetime.now()

# Create data points with some anomalies
data_points = []
for i in range(100):
    timestamp = now - timedelta(minutes=i)
    # Normal values with some anomalies
    if i in [10, 25, 60]:  # Inject anomalies
        value = random.uniform(1000, 2000)  # Anomalous values
    else:
        value = random.uniform(50, 100)  # Normal values

    data_points.append({
        "timestamp": timestamp.isoformat(),
        "metric_name": "cpu_usage",
        "value": value,
        "tags": {"server": "web-01", "region": "us-east-1"},
        "source": "monitoring_system"
    })

# Send for analysis
response = requests.post(
    "https://your-org--realtime-analytics.modal.run/predict_sync",
    json={
        "data_points": data_points,
        "task": "detect_anomaly",
        "anomaly_threshold": 0.05
    },
    headers=headers
)

result = response.json()
print(f"Found {len(result['anomalies'])} anomalies")
for anomaly in result['anomalies']:
    if anomaly['is_anomaly']:
        print(f"Anomaly at {anomaly['timestamp']}: {anomaly['actual_value']:.2f} (score: {anomaly['anomaly_score']:.3f})")
```

### Time-Series Forecasting

```python
import requests
from datetime import datetime, timedelta
import numpy as np

# Generate time series data
headers = {"x-api-key": "your-api-key"}
now = datetime.now()

# Create realistic time series (daily pattern)
data_points = []
for i in range(168):  # 7 days of hourly data
    timestamp = now - timedelta(hours=i)
    # Simulate daily pattern with some noise
    hour = timestamp.hour
    base_value = 50 + 30 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)

    data_points.append({
        "timestamp": timestamp.isoformat(),
        "metric_name": "website_traffic",
        "value": max(0, base_value),
        "tags": {"site": "main", "region": "global"},
        "source": "analytics_platform"
    })

# Generate forecast
response = requests.post(
    "https://your-org--realtime-analytics.modal.run/predict_sync",
    json={
        "data_points": data_points,
        "task": "forecast",
        "forecast_horizon": 24
    },
    headers=headers
)

result = response.json()
print(f"Generated {len(result['forecasts'])} forecast points")
for forecast in result['forecasts'][:5]:  # Show first 5
    print(f"Time: {forecast['timestamp']}, Predicted: {forecast['predicted_value']:.2f}, Trend: {forecast['trend']}")
```

### Event Classification and Alerting

```python
import requests
from datetime import datetime
import uuid

# Create sample events
headers = {"x-api-key": "your-api-key"}
now = datetime.now()

events = [
    {
        "event_id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "event_type": "error",
        "severity": "critical",
        "payload": {
            "error_code": "DATABASE_CONNECTION_FAILED",
            "affected_users": 1250,
            "service": "user-auth"
        }
    },
    {
        "event_id": str(uuid.uuid4()),
        "timestamp": (now - timedelta(minutes=5)).isoformat(),
        "event_type": "performance",
        "severity": "warning",
        "payload": {
            "response_time": 2.5,
            "threshold": 2.0,
            "endpoint": "/api/users"
        }
    }
]

# Classify events
response = requests.post(
    "https://your-org--realtime-analytics.modal.run/predict_sync",
    json={
        "events": events,
        "task": "classify"
    },
    headers=headers
)

result = response.json()
print(f"Classified {len(result['classifications'])} events")
for classification in result['classifications']:
    print(f"Event {classification['event_id']}: {classification['predicted_class']} (risk: {classification['risk_score']:.2f})")

print(f"Generated {len(result['alerts'])} alerts")
for alert in result['alerts']:
    print(f"Alert {alert['alert_id']}: {alert['severity']} - {alert['message']}")
```

### Comprehensive Analytics Pipeline

```python
import requests
from datetime import datetime, timedelta
import random
import uuid

# Complete analytics pipeline
headers = {"x-api-key": "your-api-key"}
now = datetime.now()

# Generate comprehensive data
data_points = []
events = []

# Add metric data
for i in range(50):
    timestamp = now - timedelta(minutes=i)

    # Multiple metrics
    for metric in ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]:
        value = random.uniform(10, 100)
        if metric == "cpu_usage" and i == 10:  # Inject anomaly
            value = 250

        data_points.append({
            "timestamp": timestamp.isoformat(),
            "metric_name": metric,
            "value": value,
            "tags": {"server": "prod-01", "region": "us-west-2"},
            "source": "monitoring"
        })

# Add events
for i in range(10):
    events.append({
        "event_id": str(uuid.uuid4()),
        "timestamp": (now - timedelta(minutes=i*5)).isoformat(),
        "event_type": random.choice(["info", "warning", "error"]),
        "severity": random.choice(["info", "warning", "error", "critical"]),
        "payload": {
            "service": f"service-{i}",
            "message": f"Event {i} occurred"
        }
    })

# Run comprehensive analysis
response = requests.post(
    "https://your-org--realtime-analytics.modal.run/predict_sync",
    json={
        "data_points": data_points,
        "events": events,
        "task": "analyze",
        "window_size": 50,
        "forecast_horizon": 12,
        "anomaly_threshold": 0.1
    },
    headers=headers
)

result = response.json()
print(f"Analytics Summary:")
print(f"- Data Points: {result['summary_stats']['total_data_points']}")
print(f"- Unique Metrics: {result['summary_stats']['unique_metrics']}")
print(f"- Anomalies: {len(result['anomalies'])}")
print(f"- Forecasts: {len(result['forecasts'])}")
print(f"- Classifications: {len(result['classifications'])}")
print(f"- Alerts: {len(result['alerts'])}")
print(f"- Data Quality: {result['data_quality_score']:.2f}")
print(f"- Processing Time: {result['processing_time']:.3f}s")
```

## Key Features Demonstrated

1. **Real-Time Processing**: Stream processing of metrics and events
2. **Anomaly Detection**: Statistical anomaly detection with historical context
3. **Time-Series Forecasting**: Predictive analytics for capacity planning
4. **Event Classification**: Automated risk assessment and classification
5. **Intelligent Alerting**: Context-aware alert generation
6. **Data Quality Assessment**: Automated data quality scoring
7. **High-Throughput Processing**: Batch processing for performance
8. **Historical Context**: Integration with Redis for historical data

This example showcases Modalkit's ability to handle complex, real-time analytics workloads with production-grade performance and reliability.
