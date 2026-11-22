import json
import joblib
import pandas as pd

bundle = joblib.load("api/app/models/xgboost_v4_bundle.joblib")
feature_cols = bundle["feature_cols"]

df = pd.read_csv("data/features_mafaulda_v2.csv")

# Seleccionar una fila que tenga completas las 78 características
valid_rows = df.dropna(subset=feature_cols)

if valid_rows.empty:
    raise ValueError("No se encontró una fila con las 78 características completas.")

row = valid_rows.iloc[0]

payload = {
    "features": {
        col: float(row[col])
        for col in feature_cols
    }
}

with open("api/tests/prediction_request.json", "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2)

print("Solicitud generada correctamente.")
print("Número de características:", len(payload["features"]))
print("Archivo: api/tests/prediction_request.json")