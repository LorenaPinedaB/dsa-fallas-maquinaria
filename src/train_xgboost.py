
"""
Entrenamiento XGBoost - Proyecto Diagnóstico de Fallas
Dataset: MAFAULDA V1 (características temporales)
Responsable: Daniel
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features_mafaulda.csv"
OUTPUT_DIR = ROOT / "reports" / "daniel"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

EXPERIMENT_NAME = "fallas-maquinaria"
RUN_NAME = "xgb_estratificado_v1"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not TRACKING_URI:
    raise ValueError("No se encontró MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# =====================================================
# 2. CARGA Y VALIDACIÓN
# =====================================================

print("\n" + "=" * 55)
print("CARGA DEL DATASET")
print("=" * 55)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No existe: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dimensiones:", df.shape)


# =====================================================
# 3. CONSTRUCCIÓN DE LAS 10 CLASES
# =====================================================

df["target_10"] = df["fault_type"].astype(str)

bearing_faults = [
    "ball_fault",
    "cage_fault",
    "outer_race"
]

mask = df["fault_type"].isin(bearing_faults)

if df.loc[mask, "bearing_position"].isna().any():
    raise ValueError("Hay fallas de rodamiento sin bearing_position")

df.loc[mask, "target_10"] = (
    df.loc[mask, "bearing_position"].astype(str)
    + "_"
    + df.loc[mask, "fault_type"].astype(str)
)

class_counts = df["target_10"].value_counts()

print("\nDistribución de clases:")
print(class_counts)

if df["target_10"].nunique() != 10:
    raise ValueError("Se esperaban exactamente 10 clases")

if class_counts.min() < 2:
    raise ValueError("Hay clases insuficientes para estratificar")


# =====================================================
# 4. SELECCIÓN DE CARACTERÍSTICAS
# =====================================================

# Se utilizan exclusivamente las características
# estadísticas de los seis canales de vibración.

feature_cols = [
    col for col in df.columns
    if col.startswith(("underhang_", "overhang_"))
    and col not in ["underhang", "overhang"]
]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# Tratamiento de valores infinitos
X = X.replace([np.inf, -np.inf], np.nan)

# Eliminar variables completamente vacías
X = X.dropna(axis=1, how="all")

# Imputación de faltantes mediante mediana
# calculada exclusivamente con entrenamiento
if len(feature_cols) != 60:
    raise ValueError(
        f"Se esperaban 60 características. "
        f"Se encontraron {len(feature_cols)}."
    )

y_raw = df["target_10"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

print("\nNúmero de características:", X.shape[1])
print("Número de clases:", len(encoder.classes_))


# =====================================================
# 5. PARTICIÓN ESTRATIFICADA 80/20
# =====================================================

indices = np.arange(len(df))

idx_train, idx_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_raw = X.iloc[idx_train]
X_test_raw = X.iloc[idx_test]

# Imputación sin utilizar información del test
medians = X_train_raw.median()

X_train = X_train_raw.fillna(medians)
X_test = X_test_raw.fillna(medians)

y_train = y[idx_train]
y_test = y[idx_test]

print("\nEntrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)


# Guardar manifiesto para compartir la misma partición
# con Brian en la comparación de modelos.

manifest = pd.DataFrame({
    "filepath": df.iloc[indices]["filepath"].values,
    "target_10": y_raw.values,
    "partition": np.where(
        np.isin(indices, idx_test),
        "test",
        "train"
    )
})

manifest_path = OUTPUT_DIR / "split_manifest_xgb_v1.csv"
manifest.to_csv(manifest_path, index=False)


# =====================================================
# 6. CONFIGURACIÓN XGBOOST
# =====================================================

params = {
    "objective": "multi:softprob",
    "num_class": 10,
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist"
}


# =====================================================
# 7. ENTRENAMIENTO Y REGISTRO MLFLOW
# =====================================================

print("\nIniciando entrenamiento XGBoost...")

with mlflow.start_run(run_name=RUN_NAME):

    mlflow.set_tags({
        "integrante": "Daniel",
        "modelo": "XGBoost",
        "version_datos": "V1_temporal",
        "particion": "estratificada_80_20",
        "objetivo": "fault_type_10_clases"
    })

    mlflow.log_params(params)

    mlflow.log_param("dataset_version", "V1")
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_classes", len(encoder.classes_))
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_train", len(idx_train))
    mlflow.log_param("n_test", len(idx_test))

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    recalls = recall_score(
        y_test,
        y_pred,
        average=None,
        labels=np.arange(len(encoder.classes_)),
        zero_division=0
    )

    print("\nRESULTADOS")
    print("-" * 45)
    print("Accuracy:", round(accuracy, 4))
    print("F1 macro:", round(f1_macro, 4))
    print("Balanced accuracy:", round(balanced_acc, 4))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("balanced_accuracy", balanced_acc)

    for class_name, recall in zip(encoder.classes_, recalls):
        mlflow.log_metric(
            f"recall_{class_name}",
            float(recall)
        )

    # Reporte de clasificación
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = OUTPUT_DIR / "classification_report_xgb_v1.csv"
    report_df.to_csv(report_path)

    mlflow.log_artifact(str(report_path))

    # Matriz de confusión
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_))
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title("Matriz de confusión - XGBoost V1")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = OUTPUT_DIR / "confusion_matrix_xgb_v1.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    mlflow.log_artifact(str(cm_path))

    # Importancia de variables
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_path = OUTPUT_DIR / "feature_importance_xgb_v1.csv"
    importance.to_csv(importance_path, index=False)

    mlflow.log_artifact(str(importance_path))
    mlflow.log_artifact(str(manifest_path))

    # Modelo
    mlflow.xgboost.log_model(
        model,
        artifact_path="xgboost_model"
    )

    # Resumen
    results = {
        "model": "XGBoost",
        "version": "V1",
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "balanced_accuracy": float(balanced_acc),
        "n_features": int(X.shape[1]),
        "n_train": len(idx_train),
        "n_test": len(idx_test)
    }

    results_path = OUTPUT_DIR / "results_xgb_v1.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    mlflow.log_artifact(str(results_path))

    print("\nExperimento registrado en MLflow.")
    print("Run ID:", mlflow.active_run().info.run_id)


print("\n" + "=" * 55)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 55)
