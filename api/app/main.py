
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# CONFIGURACIÓN Y CARGA DEL MODELO
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_v4_bundle.joblib"

bundle = None
model = None
imputer = None
feature_cols = []
class_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle, model, imputer, feature_cols, class_names

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encontró el modelo: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_cols = bundle["feature_cols"]
    class_names = bundle["class_names"]

    if len(feature_cols) != 78:
        raise RuntimeError(
            f"Se esperaban 78 características y se encontraron {len(feature_cols)}"
        )

    if len(class_names) != 10:
        raise RuntimeError(
            f"Se esperaban 10 clases y se encontraron {len(class_names)}"
        )

    print("Modelo XGBoost V4 cargado correctamente.")
    print(f"Características: {len(feature_cols)}")
    print(f"Clases: {len(class_names)}")

    yield


app = FastAPI(
    title="API de diagnóstico de fallas en maquinaria",
    description="Servicio de inferencia basado en XGBoost V4 y 78 características de vibración.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# ESQUEMA DE ENTRADA
# --------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="Diccionario con las 78 características del modelo XGBoost V4."
    )


# --------------------------------------------------
# ENDPOINT DE SALUD
# --------------------------------------------------

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible."
        )

    return {
        "status": "healthy",
        "model_version": bundle["version"],
        "n_features": len(feature_cols),
        "n_classes": len(class_names),
    }


# --------------------------------------------------
# ENDPOINT DE PREDICCIÓN
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible."
        )

    received_features = request.features

    expected = set(feature_cols)
    received = set(received_features.keys())

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Las características recibidas no coinciden con las requeridas por V4.",
                "missing_features": missing,
                "unexpected_features": extra,
                "expected_count": len(feature_cols),
                "received_count": len(received_features),
            },
        )

    try:
        # Mantener exactamente el orden usado durante el entrenamiento.
        input_df = pd.DataFrame(
            [[received_features[col] for col in feature_cols]],
            columns=feature_cols,
        )

        # Aplicar el imputador incluido en el bundle.
        transformed = imputer.transform(input_df)

        # Recuperar los nombres de las variables.
        transformed_df = pd.DataFrame(
            transformed,
            columns=feature_cols,
        )

        prediction = model.predict(transformed_df)[0]
        probabilities = model.predict_proba(transformed_df)[0]

        # Convertir el código numérico a su nombre de falla
        predicted_index = int(prediction)

        if predicted_index < 0 or predicted_index >= len(class_names):
            raise ValueError(f"Código de clase desconocido: {predicted_index}")

        predicted_class = class_names[predicted_index]

        # Asociar cada probabilidad con su clase correspondiente
        probability_map = {
            class_names[int(class_id)]: round(float(probability), 6)
            for class_id, probability in zip(model.classes_, probabilities)
        }
        return {
            "model_version": bundle["version"],
            "predicted_class": predicted_class,
            "probabilities": probability_map,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(error)}",
        )
