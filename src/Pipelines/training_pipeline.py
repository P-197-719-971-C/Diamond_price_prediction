
import os
import sys

import pandas as pd
import sys

from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.logger.logger import logging
from src.exception.exception import CustomException

from functools import lru_cache
sys.path.append('/Users/umrav/Desktop/My projects')
"""@st.cache_data
def load_data():
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    modeltrainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
load_data()
"""



if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)




