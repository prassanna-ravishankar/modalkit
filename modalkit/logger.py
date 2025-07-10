# In a separate file to prevent cyclic import
import logging

from pythonjsonlogger import jsonlogger

LOGGER = logging.getLogger("modalkit")
LOGGER.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter("%(message)%(levelname)", timestamp=True)
logHandler.setFormatter(formatter)

LOGGER.addHandler(logHandler)
