from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from modalkit.iomodel import InferenceOutputModel


class InferencePipeline:
    """
    Base class for an inference pipeline. Subclasses should implement preprocess, predict, and postprocess methods.
    This exists for a single model version. Downstream app layers (like modal) will decide how to manage multiple.
    """

    def __init__(
        self,
        model_name: str,
        all_model_data_folder: str,
        common_settings: dict,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        """
        Initializes the InferencePipeline class.

        Args:
            model_name (str): Name of the model
            all_model_data_folder (str): Path to the folder containing all model data
            common_settings (dict): Common settings shared across models
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.common_settings = common_settings
        self.model_name = model_name
        self.all_model_data_folder = all_model_data_folder

    def on_volume_reload(self):
        """
        Hook method called after a volume reload occurs.

        This method is called by the Modal app layer after volumes have been reloaded.
        Subclasses can override this method to perform any necessary actions after
        a volume reload, such as reloading models or updating cached data.

        By default, this method does nothing.
        """
        pass

    def run_inference(self, input_list: list[BaseModel]) -> list[InferenceOutputModel]:
        """
        Runs the full inference pipeline: preprocess -> predict -> postprocess.

        Args:
            input_list (list[BaseModel]): A list of input messages to the inference pipeline.

        Returns:
            list[InferenceOutputModel]: The list of final processed results after inference.
        """
        preprocessed_data = self.preprocess(input_list)
        raw_output = self.predict(input_list, preprocessed_data)
        result = self.postprocess(input_list, raw_output)
        return result

    @abstractmethod
    def preprocess(self, input_list: list[BaseModel]) -> dict:
        """
        Prepares the input data for the model.

        Args:
            input_list (list[BaseModel]): The list of input data to be preprocessed.

        Returns:
            dict: The preprocessed data.
        """
        pass

    @abstractmethod
    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        """
        Performs the prediction using the model.

        Args:
            input_list (list[BaseModel]): The list of original input data.
            preprocessed_data (dict): The preprocessed data.

        Returns:
            Any: The raw output from the model.
        """
        pass

    @abstractmethod
    def postprocess(self, input_list: list[BaseModel], raw_output: dict) -> list[InferenceOutputModel]:
        """
        Processes the raw output from the model into usable results.

        Args:
            input_list (list[BaseModel]): The list of original input data.
            raw_output (dict): The raw output from the model.

        Returns:
            list[InferenceOutputModel]: The list of final processed results.
        """
        pass
