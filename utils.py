""" utility functions """

import os
import datetime
import yaml
import logging
from config import Config

# Get logger for this module
logger = logging.getLogger(__name__)

# Save data to YAML file
def data_dump(what, where = Config.STATUS_PATH):
  try:
    with open(where, "w", encoding = "utf-8") as f:
      yaml.dump(what, f, allow_unicode = True, sort_keys = False)
    logger.debug(f"Status saved to {where}.yaml")
  except (IOError, OSError) as e:
    # Log the full error with traceback
    logger.error(f"File {where} write failed: {str(e)}")
    if (Config.mode == "dev"):
      logger.error(f"{traceback.format_exc()}")
    raise

# Load data from YAML file
def data_load(where = Config.STATUS_PATH):
  try:
    with open(where, "r", encoding = "utf-8") as f:
      loaded_data = yaml.safe_load(f)
    logger.debug(f"Status loaded from {where}")
    return loaded_data
  except FileNotFoundError:
    logger.info(f"File {where} not found, returning empty data")
    return {}  # Return empty dict for compatibility
  except (IOError, OSError) as e:
    # Log the full error with traceback
    logger.error(f"File read failed: {str(e)}")
    if (Config.mode == "dev"):
      logger.error(f"{traceback.format_exc()}")
    raise
