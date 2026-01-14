import os
import sys

# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation  
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'Data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            data_path = os.path.join('notebooks', 'data', 'dftrain.csv')
            df = pd.read_csv(data_path)
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    datatransformation = DataTransformation()
    train_arr,test_arr,_=datatransformation.initiate_data_transformation(train_data_path, test_data_path)

    ModelTrainer=ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr, test_arr))
    