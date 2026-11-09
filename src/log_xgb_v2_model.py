import os
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models" / "daniel" / "xgboost_v2_bundle.joblib"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise EnvironmentError("MLFLOW_TRACKING_URI no está configurada.")

if not BUNDLE_PATH.exists():
    raise FileNotFoundError(BUNDLE_PATH)

# Recuperar el modelo ya entrenado: no se ejecuta nuevamente XGBoost.
bundle = joblib.load(BUNDLE_PATH)

pipeline = bundle["pipeline"]
encoder = bundle["label_encoder"]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("fallas-maquinaria")

trusted_types = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]

with mlflow.start_run(run_name="xgb_v2_model_recovery") as run:

    mlflow.set_tag("purpose", "Registro del modelo previamente entrenado")
    mlflow.set_tag("source_run", "8cf29314cc2642948fa5b51ba22c142a")

    # Autorizar únicamente los tipos identificados en la excepción.
    mlflow.sklearn.log_model(
        pipeline,
        name="xgboost_pipeline",
        skops_trusted_types=trusted_types,
    )

    # Registrar también el bundle completo, incluido LabelEncoder.
    mlflow.log_artifact(str(BUNDLE_PATH))

    mlflow.log_dict(
        {
            "classes": encoder.classes_.tolist(),
            "features": bundle["feature_columns"],
            "target_definition": bundle["target_definition"],
        },
        "inference_schema.json",
    )

    print("Modelo registrado correctamente.")
    print("Run ID:", run.info.run_id)
    print("MLflow:", tracking_uri)