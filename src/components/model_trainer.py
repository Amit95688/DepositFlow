import os
import sys
from dataclasses import dataclass
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd


# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.exception import CustomException
from src.logger.logger import logging
from src.utils import save_object
# no direct dependency on DataTransformation artifacts required here

@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Model Trainer method starts")
        try:
            model_scores = []
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ensure arrays
            X_train = np.asarray(X_train, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            models = [
                ("XGB", xgb.XGBClassifier(eval_metric='logloss')),
                ("LGBM", lgbm.LGBMClassifier()),
            ]

            trained_models = {}

            # normalize/validate y to binary 0/1 (support 'yes'/'no' or 0/1)
            def _convert_y(arr):
                s = pd.Series(arr)
                # try numeric conversion first
                num = pd.to_numeric(s, errors='coerce')
                if num.notna().all():
                    return num.astype(int).to_numpy()
                # map common string labels
                mapped = s.astype(str).str.strip().str.lower().map({
                    'yes': 1, 'y': 1, 'true': 1, '1': 1,
                    'no': 0, 'n': 0, 'false': 0, '0': 0
                })
                # fallback to numeric where possible
                mapped = mapped.fillna(pd.to_numeric(s, errors='coerce'))
                mapped = mapped.fillna(0).astype(int)
                return mapped.to_numpy()

            y_train_proc = _convert_y(y_train)
            y_test_proc = _convert_y(y_test)

            # If training labels contain only one class, abort with helpful message
            if len(np.unique(y_train_proc)) < 2:
                raise CustomException(f"Training labels contain a single class: {np.unique(y_train_proc)}. Cannot train binary classifier.", sys)

            for model_name, model in models:
                logging.info(f"Training {model_name}")

                try:
                    model.fit(X_train, y_train_proc)
                except Exception as me:
                    logging.error(f"Training failed for {model_name}: {me}")
                    # skip this model on failure
                    continue

                if hasattr(model, "predict_proba"):
                    y_pred_probs = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_probs = model.predict(X_test)
                y_pred = model.predict(X_test)

                try:
                    auc = float(roc_auc_score(y_test_proc, y_pred_probs))
                except Exception:
                    auc = None
                acc = float(accuracy_score(y_test_proc, y_pred))
                f1 = float(f1_score(y_test_proc, y_pred))
                trained_models[model_name] = model

                # use -1.0 for missing AUC so sorting works
                auc_for_sort = auc if auc is not None else -1.0
                model_scores.append({
                    "name": model_name,
                    "auc": float(auc_for_sort),
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "raw_auc": auc
                })

                logging.info(f"{model_name} AUC: {auc}, Accuracy: {acc}, F1: {f1}")

            # select best model by AUC
            best = sorted(model_scores, key=lambda x: x["auc"], reverse=True)[0]
            best_name = best["name"]
            best_model = trained_models[best_name]

            logging.info(f"Best model: {best_name} with AUC: {best['auc']}")

            # save best model
            save_path = os.path.join(os.path.dirname(self.model_trainer_config.model_obj_file_path), f"model.pkl")
            save_object(file_path=save_path, obj=best_model)
            logging.info(f"Saved best model to {save_path}")

            logging.info("Model Trainer method ends")
            return save_path

        except Exception as e:
            raise CustomException(e, sys)