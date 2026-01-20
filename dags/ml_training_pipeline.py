"""
Airflow DAG for ML Pipeline: Data Ingestion → Transformation → Model Training (PyTorch) + DVC Integration
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os
import sys

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Default arguments
default_args = {
    'owner': 'ml_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
}

# DAG definition
dag = DAG(
    'ml_pipeline_dvc_pytorch',
    default_args=default_args,
    description='DVC-based ML Pipeline: Ingestion → Transformation → Model Training (PyTorch + HPO)',
    schedule='@weekly',
    catchup=False,
    tags=['ml', 'pytorch', 'dvc', 'pipeline']
)

# Base path for logs and project
PROJECT_ROOT = "/app"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# -------------------------------
# Stage 1: Data Ingestion
# -------------------------------
data_ingestion_task = BashOperator(
    task_id="data_ingestion",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_ingestion >> {LOGS_DIR}/data_ingestion.log 2>&1",
    dag=dag,
)

# -------------------------------
# Stage 2: Data Transformation
# -------------------------------
data_transformation_task = BashOperator(
    task_id="data_transformation",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_transformation >> {LOGS_DIR}/data_transformation.log 2>&1",
    dag=dag,
)

# -------------------------------
# Stage 3: Model Training
# -------------------------------
model_training_task = BashOperator(
    task_id="model_trainer",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s model_trainer >> {LOGS_DIR}/model_trainer.log 2>&1",
    dag=dag,
)

# -------------------------------
# Stage 4: Push Artifacts to DVC Remote (Optional)
# -------------------------------
push_artifacts_task = BashOperator(
    task_id="push_artifacts",
    bash_command=f"cd {PROJECT_ROOT} && dvc push >> {LOGS_DIR}/dvc_push.log 2>&1",
    dag=dag,
)

# -------------------------------
# Set execution order
# -------------------------------
data_ingestion_task >> data_transformation_task >> model_training_task >> push_artifacts_task