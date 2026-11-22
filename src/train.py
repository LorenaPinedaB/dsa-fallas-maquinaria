import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# 1. CONFIGURACIÓN DE RUTAS 
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features_mafaulda_v2.csv"
MANIFEST_PATH = ROOT / "reports" / "daniel" / "split_manifest_xgb_v1.csv"
REPORT_DIR = ROOT / "reports" / "brian"

RANDOM_STATE = 42
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. CARGA Y HOMOLOGACIÓN DE DATOS (10 CLASES, 78 FEATURES)
def load_homologated_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {DATA_PATH}. Verifica que esté en la carpeta data/.")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No se encontró el manifiesto en {MANIFEST_PATH}.")

    print("Cargando dataset espectral y manifiesto oficial...")
    df = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    # Construir target de 10 
    df["target_10"] = df["fault_type"].astype(str)
    bearing_faults = ["ball_fault", "cage_fault", "outer_race"]
    mask = df["fault_type"].isin(bearing_faults)
    df.loc[mask, "target_10"] = (
        df.loc[mask, "bearing_position"].astype(str)
        + "_"
        + df.loc[mask, "fault_type"].astype(str)
    )

    # Recuperar TRAIN (1560) y TEST (391) oficiales mediante 'filepath'
    train_manifest = manifest.loc[manifest["partition"] == "train", ["filepath", "target_10"]]
    test_manifest = manifest.loc[manifest["partition"] == "test", ["filepath", "target_10"]]

    df_train = df.merge(
        train_manifest, on="filepath", how="inner", suffixes=("", "_manifest"), validate="one_to_one"
    )
    df_test = df.merge(
        test_manifest, on="filepath", how="inner", suffixes=("", "_manifest"), validate="one_to_one"
    )

    # Validar integridad de particiones y etiquetas
    for name, data, expected in [("TRAIN", df_train, len(train_manifest)), ("TEST", df_test, len(test_manifest))]:
        if len(data) != expected:
            raise ValueError(f"No coinciden los registros de {name}: {len(data)} vs {expected}.")
        if not (data["target_10"] == data["target_10_manifest"]).all():
            raise ValueError(f"Los targets de {name} no coinciden con el manifiesto.")
        data.drop(columns=["target_10_manifest"], inplace=True)

    # Seleccionar exactamente las 60 características temporales
    temporal_features = [
        c for c in df.columns
        if any(
            c == f"{channel}_{stat}"
            for channel in [
                "underhang_axial", "underhang_radial", "underhang_tangential",
                "overhang_axial", "overhang_radial", "overhang_tangential",
            ]
            for stat in [
                "mean", "std", "rms", "min", "max",
                "peak_to_peak", "peak", "skewness", "kurtosis", "crest_factor",
            ]
        )
    ]

    # Seleccionar exactamente las 18 características espectrales
    spectral_features = [
        c for c in df.columns
        if c.endswith("_order_1x_amplitude")
        or c.endswith("_order_2x_amplitude")
        or c.endswith("_order_3x_amplitude")
    ]

    feature_cols = sorted(temporal_features + spectral_features)

    if len(feature_cols) != 78:
        raise ValueError(f"Se esperaban 78 características, se encontraron {len(feature_cols)}.")

    X_train = df_train[feature_cols].copy()
    y_train = df_train["target_10"].reset_index(drop=True)

    X_test = df_test[feature_cols].copy()
    y_test = df_test["target_10"].reset_index(drop=True)

    # Imputación con mediana ajustada solo en TRAIN (protocolo XGBoost V4)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    print(f"Datos homologados listos: TRAIN={len(X_train)} | TEST={len(X_test)} | Features={len(feature_cols)} | Clases={y_train.nunique()}")
    return X_train_imp, X_test_imp, y_train, y_test, feature_cols

# 3. ENTRENAMIENTO Y REGISTRO EN MLFLOW
def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, run_name, params, feature_cols):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI no está configurada en la terminal.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fallas-maquinaria")

    with mlflow.start_run(run_name=run_name):
        # 1. Entrenar y predecir sobre TEST oficial (391 registros)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 2. Métricas homologadas con XGBoost V4
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        classes = list(model.classes_)
        rec_normal = recall_score(y_test, y_pred, average=None)[classes.index("normal")]

        # 3. Registrar Parámetros
        mlflow.log_params({
            "modelo": model_name,
            "tipo_particion": "manifiesto_fijo_xgb_v1",
            "version_datos": "v2_78_features_sin_rpm",
            "n_features_total": len(feature_cols),
            "n_classes": len(classes),
            "random_state": RANDOM_STATE,
            **params
        })

        # 4. Registrar Métricas 
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_macro": f1_mac,
            "balanced_accuracy": bal_acc,
            "recall_normal": rec_normal,
            "test_accuracy": acc,
            "test_f1_macro": f1_mac,
            "test_balanced_accuracy": bal_acc,
        })

        # 5. Guardar Reporte de Clasificación y Matriz de Confusión Normalizada
        report_txt = classification_report(y_test, y_pred, zero_division=0)
        report_file = REPORT_DIR / f"classification_report_{run_name}.txt"
        report_file.write_text(report_txt, encoding="utf-8")
        mlflow.log_artifact(str(report_file))

        fig, ax = plt.subplots(figsize=(13, 11))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, xticks_rotation="vertical", cmap="viridis", normalize="true", values_format=".2f"
        )
        plt.title(f"Matriz de Confusión (10 Clases - Homologado) — {model_name}")
        plt.tight_layout()

        cm_file = REPORT_DIR / f"matriz_confusion_{run_name}.png"
        plt.savefig(cm_file)
        mlflow.log_artifact(str(cm_file))
        plt.close()

        # 6. Registrar modelo empaquetado
        trusted = ["sklearn.tree._tree.Tree"] if "Forest" in model_name else None
        mlflow.sklearn.log_model(sk_model=model, name="modelo", skops_trusted_types=trusted)

        print(
            f"🚀 {run_name} registrado | "
            f"Accuracy: {acc:.4f} | F1-Macro: {f1_mac:.4f} | "
            f"Balanced Acc: {bal_acc:.4f} | Recall Normal: {rec_normal:.4f}"
        )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = load_homologated_data()

    # Escalado ajustado estrictamente sobre X_train para Regresión Logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- MODELO 1: REGRESIÓN LOGÍSTICA HOMOLOGADA (10 CLASES, 78 FEATURES) ---
    lr_params = {"max_iter": 1000, "C": 1.0}
    lr_model = LogisticRegression(**lr_params, random_state=RANDOM_STATE)
    train_and_log_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        model=lr_model,
        model_name="Regresion Logistica",
        run_name="lr_homologado_10clases_v4",
        params=lr_params,
        feature_cols=feature_cols,
    )

    # --- MODELO 2: RANDOM FOREST HOMOLOGADO (10 CLASES, 78 FEATURES) ---
    rf_params = {"n_estimators": 200, "max_depth": 15}
    rf_model = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    train_and_log_model(
        X_train, X_test, y_train, y_test,
        model=rf_model,
        model_name="Random Forest",
        run_name="rf_homologado_10clases_v4",
        params=rf_params,
        feature_cols=feature_cols,
    )