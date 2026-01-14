import os 
import sys
import dill
import pickle

# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger.logger import logging
import pandas as pd
import numpy as np


def save_object(file_path, obj):
    """Save object to file using dill"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}")
        raise CustomException(e, sys)


def load_object(file_path):
    """Load object from file using dill"""
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error occurred while loading object from {file_path}")
        raise CustomException(e, sys)
    