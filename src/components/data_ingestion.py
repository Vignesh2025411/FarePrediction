import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

'''from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer'''


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths"""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """Class to handle data ingestion from source and split into train/test"""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Read data from source, save raw data, and split into train/test sets
        
        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read dataset using os.path.join for cross-platform compatibility
            df = pd.read_csv(os.path.join('notebook', 'data', 'ride_fare_clean.csv'))
            logging.info('Read the dataset as dataframe')

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            # Split data into train and test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    # Initialize data ingestion
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    '''# Initialize data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Initialize and train model
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))'''