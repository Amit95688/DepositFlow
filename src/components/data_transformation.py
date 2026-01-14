import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Feature engineering with trigonometric features
            logging.info("Creating trigonometric features")
            train_df['_duration_sin'] = np.sin(2*np.pi * train_df['duration'] / 540).astype('float32')
            train_df['_duration_cos'] = np.cos(2*np.pi * train_df['duration'] / 540).astype('float32')
            train_df['_balance_log'] = (np.sign(train_df['balance']) * np.log1p(np.abs(train_df['balance']))).astype('float32')
            train_df['_balance_sin'] = np.sin(2*np.pi * train_df['balance'] / 1000).astype('float32')
            train_df['_balance_cos'] = np.cos(2*np.pi * train_df['balance'] / 1000).astype('float32')
            train_df['_age_sin'] = np.sin(2*np.pi * train_df['age'] / 10).astype('float32')
            train_df['_pdays_sin'] = np.sin(2*np.pi * train_df['pdays'] / 7).astype('float32')

            test_df['_duration_sin'] = np.sin(2*np.pi * test_df['duration'] / 540).astype('float32')
            test_df['_duration_cos'] = np.cos(2*np.pi * test_df['duration'] / 540).astype('float32')
            test_df['_balance_log'] = (np.sign(test_df['balance']) * np.log1p(np.abs(test_df['balance']))).astype('float32')
            test_df['_balance_sin'] = np.sin(2*np.pi * test_df['balance'] / 1000).astype('float32')
            test_df['_balance_cos'] = np.cos(2*np.pi * test_df['balance'] / 1000).astype('float32')
            test_df['_age_sin'] = np.sin(2*np.pi * test_df['age'] / 10).astype('float32')
            test_df['_pdays_sin'] = np.sin(2*np.pi * test_df['pdays'] / 7).astype('float32')

            # Convert target if it exists (support numeric 0/1 and 'yes'/'no' variants)
            def _convert_target(series):
                # numeric types (int/float)
                if pd.api.types.is_numeric_dtype(series):
                    vals = series.fillna(0).astype(int)
                    # ensure values are 0/1
                    vals = vals.where(vals.isin([0, 1]), 0)
                    return vals.astype('int32')
                # object / string types
                s = series.astype(str).str.strip().str.lower()
                mapping = {'yes': 1, 'y': 1, '1': 1, 'true': 1, 'no': 0, 'n': 0, '0': 0, 'false': 0}
                mapped = s.map(mapping)
                # fallback: try numeric conversion
                mapped = mapped.fillna(pd.to_numeric(s, errors='coerce'))
                mapped = mapped.fillna(0).astype(int)
                return mapped.astype('int32')

            if 'y' in train_df.columns:
                train_df['y'] = _convert_target(train_df['y'])
                if 'y' in test_df.columns:
                    test_df['y'] = _convert_target(test_df['y'])
                train_target_counts = train_df['y'].value_counts().to_dict()
                logging.info(f"Target distribution: {train_target_counts}")
                # Guard: ensure there are at least two classes in training set
                if train_df['y'].nunique() < 2:
                    raise CustomException(f"Training target has a single class: {list(train_df['y'].unique())}. Cannot train a binary classifier.", sys)
            
            logging.info(f"Features created: {[c for c in train_df.columns if c.startswith('_')]}")

            # Interaction features
            logging.info("Creating interaction features")
            if 'age' in train_df.columns and 'balance' in train_df.columns:
                train_df['_age_balance'] = (train_df['age'] * train_df['balance']).astype('float32')
                test_df['_age_balance'] = (test_df['age'] * test_df['balance']).astype('float32')
            
            if 'duration' in train_df.columns and 'campaign' in train_df.columns:
                train_df['_duration_campaign'] = (train_df['duration'] * train_df['campaign']).astype('float32')
                test_df['_duration_campaign'] = (test_df['duration'] * test_df['campaign']).astype('float32')
            
            # Ratio features
            if 'previous' in train_df.columns and 'campaign' in train_df.columns:
                train_df['_prev_campaign_ratio'] = (train_df['previous'] / (train_df['campaign'] + 1)).astype('float32')
                test_df['_prev_campaign_ratio'] = (test_df['previous'] / (test_df['campaign'] + 1)).astype('float32')
            
            # Binning features
            if 'age' in train_df.columns:
                train_df['_age_group'] = pd.cut(train_df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype('int32')
                test_df['_age_group'] = pd.cut(test_df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype('int32')
            
            logging.info("Interaction and binning features created")

            # Separate target variable
            target_col = 'y'
            if target_col in train_df.columns:
                y_train = train_df[target_col]
                train_df = train_df.drop(columns=[target_col])
            else:
                y_train = None
            
            if target_col in test_df.columns:
                y_test = test_df[target_col]
                test_df = test_df.drop(columns=[target_col])
            else:
                y_test = None

            logging.info("Building preprocessing pipeline")

            # Identify numerical and categorical columns
            numerical_cols = train_df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()
            categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
            
            logging.info(f"Numerical columns: {len(numerical_cols)}, Categorical columns: {len(categorical_cols)}")

            # Numerical pipeline: Impute missing values + Scale
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]
            )

            # Categorical pipeline: Impute + Label Encode
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent'))
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_cols),
                    ('cat', categorical_pipeline, categorical_cols)
                ]
            )

            # Fit and transform
            logging.info("Fitting preprocessing pipeline on train data")
            train_arr = preprocessor.fit_transform(train_df)
            test_arr = preprocessor.transform(test_df)

            # Label encode categorical columns after transformation
            logging.info("Label encoding categorical features")
            label_encoders = {}
            cat_start_idx = len(numerical_cols)
            for i, col in enumerate(categorical_cols):
                le = LabelEncoder()
                train_arr[:, cat_start_idx + i] = le.fit_transform(train_arr[:, cat_start_idx + i].astype(str))
                test_arr[:, cat_start_idx + i] = le.transform(test_arr[:, cat_start_idx + i].astype(str))
                label_encoders[col] = le

            # Add target back if exists
            if y_train is not None:
                train_arr = np.c_[train_arr, np.array(y_train)]
            if y_test is not None:
                test_arr = np.c_[test_arr, np.array(y_test)]
            
            logging.info("Preprocessing pipeline completed")

            # Save preprocessing objects
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            preprocessing_obj = {
                'preprocessor': preprocessor,
                'label_encoders': label_encoders,
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols
            }

            logging.info("Preprocessing object saved")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)

