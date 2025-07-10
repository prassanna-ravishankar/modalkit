from pydantic import BaseModel

from modalkit.iomodel import AsyncInputModel, AsyncOutputModel


def test_async_output_model():
    output = AsyncOutputModel(job_id="test-job")
    assert output.job_id == "test-job"


def test_async_input_model():
    class TestMessage(BaseModel):
        data: str

    input_model = AsyncInputModel(
        message=TestMessage(data="test"),
        success_queue="success-queue",
        failure_queue="failure-queue",
        meta={"key": "value"},
    )

    assert input_model.message.data == "test"
    assert input_model.success_queue == "success-queue"
    assert input_model.failure_queue == "failure-queue"
    assert input_model.meta == {"key": "value"}
