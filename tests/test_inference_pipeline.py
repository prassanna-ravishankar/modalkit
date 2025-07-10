"""
Tests for InferencePipeline behavior.

This module tests the core ML inference pipeline functionality, focusing on:
- Pipeline execution behavior
- Data flow through preprocessing, prediction, and postprocessing stages
- Error handling and edge cases
- Hook method integration
"""

from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from modalkit.inference import InferencePipeline
from modalkit.iomodel import InferenceOutputModel


class SentimentInput(BaseModel):
    """Input model for sentiment analysis pipeline"""

    text: str
    language: str = "en"


class SentimentOutput(InferenceOutputModel):
    """Output model for sentiment analysis results"""

    sentiment: str
    confidence: float


class TextProcessingPipeline(InferencePipeline):
    """Concrete implementation of InferencePipeline for testing"""

    def __init__(self, model_name: str, model_folder: str, common_settings: dict):
        super().__init__(model_name, model_folder, common_settings)
        self.preprocessing_called = False
        self.prediction_called = False
        self.postprocessing_called = False

    def preprocess(self, input_list: list[BaseModel]) -> dict:
        """Preprocesses text input by cleaning and tokenizing"""
        self.preprocessing_called = True
        processed_texts = []
        for input_item in input_list:
            # Simulate text cleaning and tokenization
            clean_text = input_item.text.strip().lower()
            processed_texts.append(clean_text)

        return {"processed_texts": processed_texts, "batch_size": len(input_list)}

    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        """Simulates sentiment prediction using a mock model"""
        self.prediction_called = True
        predictions = []

        for text in preprocessed_data["processed_texts"]:
            # Mock sentiment prediction based on simple keywords
            if "good" in text or "great" in text or "excellent" in text:
                sentiment = "positive"
                confidence = 0.9
            elif "bad" in text or "terrible" in text or "awful" in text:
                sentiment = "negative"
                confidence = 0.85
            else:
                sentiment = "neutral"
                confidence = 0.6

            predictions.append({"sentiment": sentiment, "confidence": confidence})

        return {"predictions": predictions}

    def postprocess(self, input_list: list[BaseModel], raw_output: dict) -> list[InferenceOutputModel]:
        """Converts raw predictions to structured output format"""
        self.postprocessing_called = True
        results = []

        for prediction in raw_output["predictions"]:
            output = SentimentOutput(
                status="success", sentiment=prediction["sentiment"], confidence=prediction["confidence"]
            )
            results.append(output)

        return results


class ErrorPronePipeline(InferencePipeline):
    """Pipeline that simulates errors in different stages"""

    def __init__(self, error_stage: Optional[str] = None):
        import tempfile

        super().__init__("error-model", tempfile.gettempdir(), {})
        self.error_stage = error_stage

    def preprocess(self, input_list: list[BaseModel]) -> dict:
        if self.error_stage == "preprocess":
            raise ValueError("Preprocessing failed")
        return {"data": ["processed"]}

    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        if self.error_stage == "predict":
            raise RuntimeError("Model prediction failed")
        return {"predictions": ["result"]}

    def postprocess(self, input_list: list[BaseModel], raw_output: dict) -> list[InferenceOutputModel]:
        if self.error_stage == "postprocess":
            raise ValueError("Postprocessing failed")
        return [InferenceOutputModel(status="success")]


class TestInferencePipelineBehavior:
    """Test suite focusing on InferencePipeline behavior and data flow"""

    @pytest.fixture
    def sample_inputs(self):
        """Provides sample inputs for testing"""
        return [
            SentimentInput(text="This is a great product!", language="en"),
            SentimentInput(text="This is terrible quality", language="en"),
            SentimentInput(text="It's okay, nothing special", language="en"),
        ]

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Creates a test pipeline instance"""
        return TextProcessingPipeline(
            model_name="sentiment-analyzer-v1",
            model_folder=str(tmp_path / "models"),
            common_settings={"cache_size": 1000, "timeout": 30},
        )

    def test_pipeline_initializes_with_correct_attributes(self, pipeline):
        """Pipeline should initialize with provided configuration"""
        assert pipeline.model_name == "sentiment-analyzer-v1"
        assert "models" in pipeline.all_model_data_folder
        assert pipeline.common_settings["cache_size"] == 1000
        assert pipeline.common_settings["timeout"] == 30

    def test_pipeline_executes_complete_inference_workflow(self, pipeline, sample_inputs):
        """Pipeline should execute all stages in correct order"""
        results = pipeline.run_inference(sample_inputs)

        # Verify all stages were called
        assert pipeline.preprocessing_called
        assert pipeline.prediction_called
        assert pipeline.postprocessing_called

        # Verify results structure
        assert len(results) == 3
        assert all(isinstance(result, SentimentOutput) for result in results)
        assert all(result.status == "success" for result in results)

    def test_pipeline_processes_positive_sentiment_correctly(self, pipeline):
        """Pipeline should correctly identify positive sentiment"""
        positive_input = [SentimentInput(text="This is an excellent product!")]

        results = pipeline.run_inference(positive_input)

        assert len(results) == 1
        assert results[0].sentiment == "positive"
        assert results[0].confidence > 0.8

    def test_pipeline_processes_negative_sentiment_correctly(self, pipeline):
        """Pipeline should correctly identify negative sentiment"""
        negative_input = [SentimentInput(text="This is a terrible experience")]

        results = pipeline.run_inference(negative_input)

        assert len(results) == 1
        assert results[0].sentiment == "negative"
        assert results[0].confidence > 0.8

    def test_pipeline_processes_neutral_sentiment_correctly(self, pipeline):
        """Pipeline should handle neutral sentiment appropriately"""
        neutral_input = [SentimentInput(text="The weather is fine today")]

        results = pipeline.run_inference(neutral_input)

        assert len(results) == 1
        assert results[0].sentiment == "neutral"
        assert 0.5 <= results[0].confidence <= 0.7

    def test_pipeline_handles_batch_processing(self, pipeline, sample_inputs):
        """Pipeline should process multiple inputs in a single batch"""
        results = pipeline.run_inference(sample_inputs)

        assert len(results) == 3
        sentiments = [result.sentiment for result in results]
        assert "positive" in sentiments
        assert "negative" in sentiments
        assert "neutral" in sentiments

    def test_pipeline_handles_empty_input_list(self, pipeline):
        """Pipeline should gracefully handle empty input lists"""
        results = pipeline.run_inference([])

        assert results == []
        assert pipeline.preprocessing_called
        assert pipeline.prediction_called
        assert pipeline.postprocessing_called

    def test_pipeline_propagates_preprocessing_errors(self):
        """Pipeline should propagate errors that occur during preprocessing"""
        error_pipeline = ErrorPronePipeline(error_stage="preprocess")

        with pytest.raises(ValueError, match="Preprocessing failed"):
            error_pipeline.run_inference([SentimentInput(text="test")])

    def test_pipeline_propagates_prediction_errors(self):
        """Pipeline should propagate errors that occur during prediction"""
        error_pipeline = ErrorPronePipeline(error_stage="predict")

        with pytest.raises(RuntimeError, match="Model prediction failed"):
            error_pipeline.run_inference([SentimentInput(text="test")])

    def test_pipeline_propagates_postprocessing_errors(self):
        """Pipeline should propagate errors that occur during postprocessing"""
        error_pipeline = ErrorPronePipeline(error_stage="postprocess")

        with pytest.raises(ValueError, match="Postprocessing failed"):
            error_pipeline.run_inference([SentimentInput(text="test")])

    def test_volume_reload_hook_is_callable(self, pipeline):
        """Volume reload hook should be callable without errors by default"""
        # Should not raise any exceptions
        pipeline.on_volume_reload()

    def test_volume_reload_hook_can_be_overridden(self):
        """Volume reload hook should be overridable for custom behavior"""

        class CustomPipeline(InferencePipeline):
            def __init__(self):
                import tempfile

                super().__init__("test", tempfile.gettempdir(), {})
                self.reload_called = False

            def on_volume_reload(self):
                self.reload_called = True

            def preprocess(self, input_list):
                return {}

            def predict(self, input_list, preprocessed_data):
                return {}

            def postprocess(self, input_list, raw_output):
                return []

        pipeline = CustomPipeline()
        pipeline.on_volume_reload()

        assert pipeline.reload_called

    def test_pipeline_data_flow_integrity(self, pipeline, sample_inputs):
        """Data should flow correctly between pipeline stages"""
        # Mock the stages to verify data flow
        with (
            patch.object(pipeline, "preprocess", wraps=pipeline.preprocess) as mock_preprocess,
            patch.object(pipeline, "predict", wraps=pipeline.predict) as mock_predict,
            patch.object(pipeline, "postprocess", wraps=pipeline.postprocess) as mock_postprocess,
        ):
            pipeline.run_inference(sample_inputs)

            # Verify preprocess was called with original inputs
            mock_preprocess.assert_called_once_with(sample_inputs)

            # Verify predict was called with original inputs and preprocessed data
            args, kwargs = mock_predict.call_args
            assert args[0] == sample_inputs  # original inputs
            assert "processed_texts" in args[1]  # preprocessed data

            # Verify postprocess was called with original inputs and predictions
            args, kwargs = mock_postprocess.call_args
            assert args[0] == sample_inputs  # original inputs
            assert "predictions" in args[1]  # raw predictions

    def test_pipeline_preserves_input_order(self, pipeline):
        """Pipeline should preserve the order of inputs in outputs"""
        ordered_inputs = [
            SentimentInput(text="Great product!"),  # positive
            SentimentInput(text="Terrible quality"),  # negative
            SentimentInput(text="Okay product"),  # neutral
            SentimentInput(text="Excellent service!"),  # positive
        ]

        results = pipeline.run_inference(ordered_inputs)

        assert len(results) == 4
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"
        assert results[2].sentiment == "neutral"
        assert results[3].sentiment == "positive"

    def test_pipeline_maintains_consistent_interface(self, pipeline):
        """Pipeline should maintain consistent interface regardless of input content"""
        inputs_variety = [
            [SentimentInput(text="single input")],
            [SentimentInput(text="input 1"), SentimentInput(text="input 2")],
            [],
        ]

        for inputs in inputs_variety:
            results = pipeline.run_inference(inputs)

            # Results should always be a list
            assert isinstance(results, list)
            # Each result should be an InferenceOutputModel
            assert all(isinstance(r, InferenceOutputModel) for r in results)
            # Length should match input length
            assert len(results) == len(inputs)


class TestInferencePipelineContract:
    """Test suite for pipeline contract and method signatures"""

    def test_pipeline_method_signatures_are_enforced(self):
        """Pipeline methods should be called with expected signatures"""

        class SignatureTestPipeline(InferencePipeline):
            def __init__(self):
                import tempfile

                super().__init__("test", tempfile.gettempdir(), {})
                self.preprocess_args = None
                self.predict_args = None
                self.postprocess_args = None

            def preprocess(self, input_list):
                self.preprocess_args = (input_list,)
                return {"data": "preprocessed"}

            def predict(self, input_list, preprocessed_data):
                self.predict_args = (input_list, preprocessed_data)
                return {"predictions": ["result"]}

            def postprocess(self, input_list, raw_output):
                self.postprocess_args = (input_list, raw_output)
                return [InferenceOutputModel(status="success")]

        pipeline = SignatureTestPipeline()
        test_input = [SentimentInput(text="test")]

        pipeline.run_inference(test_input)

        # Verify all methods were called with correct signatures
        assert pipeline.preprocess_args[0] == test_input
        assert pipeline.predict_args[0] == test_input
        assert pipeline.predict_args[1] == {"data": "preprocessed"}
        assert pipeline.postprocess_args[0] == test_input
        assert pipeline.postprocess_args[1] == {"predictions": ["result"]}

    def test_pipeline_can_be_extended_with_custom_logic(self):
        """Pipeline can be extended with custom initialization and behavior"""

        class CustomPipeline(InferencePipeline):
            def __init__(self, model_name, model_folder, common_settings):
                super().__init__(model_name, model_folder, common_settings)
                self.custom_config = common_settings.get("custom_param", "default")
                self.processed_count = 0

            def preprocess(self, input_list):
                self.processed_count += len(input_list)
                return {"count": len(input_list)}

            def predict(self, input_list, preprocessed_data):
                return {"predictions": ["custom"] * preprocessed_data["count"]}

            def postprocess(self, input_list, raw_output):
                return [InferenceOutputModel(status="success") for _ in raw_output["predictions"]]

        import tempfile

        pipeline = CustomPipeline("custom-model", tempfile.gettempdir(), {"custom_param": "test_value"})

        assert pipeline.custom_config == "test_value"
        assert pipeline.processed_count == 0

        results = pipeline.run_inference([SentimentInput(text="test1"), SentimentInput(text="test2")])

        assert pipeline.processed_count == 2
        assert len(results) == 2
