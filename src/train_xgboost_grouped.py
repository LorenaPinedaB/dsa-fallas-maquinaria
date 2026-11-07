
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
import mlflow

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features_mafaulda.csv"
MANIFEST_PATH = ROOT / "reports" / "daniel" / "split_manifest_xgb_v1.csv"
REPORT_DIR = ROOT / "reports" / "daniel"

RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 1500
N_JOBS = 1

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGA Y PREPARACIÓN
# ============================================================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

# Construir target de diez clases
df["target_10"] = df["fault_type"].astype(str)

bearing_faults = ["ball_fault", "cage_fault", "outer_race"]
mask = df["fault_type"].isin(bearing_faults)

df.loc[mask, "target_10"] = (
    df.loc[mask, "bearing_position"].astype(str)
    + "_"
    + df.loc[mask, "fault_type"].astype(str)
)

# Recuperar únicamente el TRAIN oficial
train_manifest = manifest.loc[
    manifest["partition"] == "train",
    ["filepath", "target_10"]
]

df_train = df.merge(
    train_manifest,
    on="filepath",
    how="inner",
    suffixes=("", "_manifest"),
    validate="one_to_one"
)

if len(df_train) != len(train_manifest):
    raise ValueError("No coinciden los registros del manifiesto.")

if not (
    df_train["target_10"] == df_train["target_10_manifest"]
).all():
    raise ValueError("Los targets no coinciden con el manifiesto.")

df_train = df_train.drop(columns=["target_10_manifest"])

# Predictores: exclusivamente las 60 características temporales
feature_cols = sorted([
    c for c in df_train.columns
    if c.startswith("underhang_") or c.startswith("overhang_")
])

if len(feature_cols) != 60:
    raise ValueError(f"Se esperaban 60 características, encontradas: {len(feature_cols)}")

X = df_train[feature_cols].copy()
y_text = df_train["target_10"].reset_index(drop=True)
groups = df_train["nominal_speed_rpm"].reset_index(drop=True)

if groups.isna().any():
    raise ValueError("Hay velocidades nulas.")

if X.isna().all().any():
    raise ValueError("Hay características completamente nulas.")

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

class_names = encoder.classes_.tolist()
n_classes = len(class_names)

print("Registros:", len(X))
print("Características:", len(feature_cols))
print("Clases:", n_classes)
print("Velocidades:", groups.nunique())


# ============================================================
# 3. FUNCIONES DE MODELO
# ============================================================

def create_model():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=MAX_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        importance_type="gain",
    )


# ============================================================
# 4. VALIDACIÓN CRUZADA AGRUPADA
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
oof_true = []
oof_pred = []
oof_records = []

print("\nINICIANDO VALIDACIÓN AGRUPADA POR VELOCIDAD")

for fold, (outer_train, outer_valid) in enumerate(
    outer_cv.split(X, y, groups), start=1
):

    X_outer_train = X.iloc[outer_train]
    X_outer_valid = X.iloc[outer_valid]

    y_outer_train = y[outer_train]
    y_outer_valid = y[outer_valid]

    groups_outer_train = groups.iloc[outer_train]
    groups_outer_valid = groups.iloc[outer_valid]

    # Comprobar separación de velocidades
    overlap = set(groups_outer_train).intersection(
        set(groups_outer_valid)
    )

    if overlap:
        raise ValueError(f"Solapamiento de velocidades en fold {fold}")

    # Validación interna también agrupada por velocidad
    inner_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE + fold
    )

    inner_train, inner_valid = next(
        inner_cv.split(
            X_outer_train,
            y_outer_train,
            groups_outer_train
        )
    )

    X_inner_train = X_outer_train.iloc[inner_train]
    X_inner_valid = X_outer_train.iloc[inner_valid]

    y_inner_train = y_outer_train[inner_train]
    y_inner_valid = y_outer_train[inner_valid]

    inner_groups_train = groups_outer_train.iloc[inner_train]
    inner_groups_valid = groups_outer_train.iloc[inner_valid]

    if set(inner_groups_train).intersection(set(inner_groups_valid)):
        raise ValueError("Solapamiento en validación interna.")

    # Imputación ajustada únicamente con entrenamiento interno
    imputer = SimpleImputer(strategy="median")

    X_inner_train_imp = imputer.fit_transform(X_inner_train)
    X_inner_valid_imp = imputer.transform(X_inner_valid)
    X_outer_valid_imp = imputer.transform(X_outer_valid)

    model = create_model()

    model.fit(
        X_inner_train_imp,
        y_inner_train,
        eval_set=[(X_inner_valid_imp, y_inner_valid)],
        verbose=False
    )

    # Evaluación externa: velocidades completamente no vistas
    pred = model.predict(X_outer_valid_imp)

    accuracy = accuracy_score(y_outer_valid, pred)
    f1 = f1_score(
        y_outer_valid, pred,
        average="macro",
        zero_division=0
    )
    balanced = balanced_accuracy_score(y_outer_valid, pred)

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "f1_macro": f1,
        "balanced_accuracy": balanced,
        "best_iteration": int(model.best_iteration),
        "train_records": len(outer_train),
        "valid_records": len(outer_valid),
        "train_speeds": groups_outer_train.nunique(),
        "valid_speeds": groups_outer_valid.nunique(),
        "group_overlap": len(overlap),
    })

    oof_true.extend(y_outer_valid.tolist())
    oof_pred.extend(pred.tolist())

    for local_idx, prediction in zip(outer_valid, pred):
        oof_records.append({
            "filepath": df_train.iloc[local_idx]["filepath"],
            "nominal_speed_rpm": groups.iloc[local_idx],
            "target_true": encoder.inverse_transform(
                [y[local_idx]]
            )[0],
            "target_pred": encoder.inverse_transform(
                [prediction]
            )[0],
            "fold": fold,
        })

    print(
        f"Fold {fold}: "
        f"Accuracy={accuracy:.4f} | "
        f"F1-macro={f1:.4f} | "
        f"Balanced accuracy={balanced:.4f} | "
        f"Best iteration={model.best_iteration}"
    )


# ============================================================
# 5. RESULTADOS AGREGADOS
# ============================================================

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    REPORT_DIR / "cv_grouped_metrics_xgb_v3.csv",
    index=False
)

oof_df = pd.DataFrame(oof_records)

oof_df.to_csv(
    REPORT_DIR / "oof_predictions_xgb_v3.csv",
    index=False
)

summary = {}

for metric in ["accuracy", "f1_macro", "balanced_accuracy"]:
    summary[f"group_cv_{metric}_mean"] = float(
        results_df[metric].mean()
    )
    summary[f"group_cv_{metric}_std"] = float(
        results_df[metric].std(ddof=1)
    )

# Métricas OOF: cada registro tiene una predicción de un modelo
# que no entrenó con su velocidad.
summary["group_oof_accuracy"] = accuracy_score(oof_true, oof_pred)
summary["group_oof_f1_macro"] = f1_score(
    oof_true, oof_pred,
    average="macro",
    zero_division=0
)
summary["group_oof_balanced_accuracy"] = balanced_accuracy_score(
    oof_true, oof_pred
)

summary["n_records"] = len(X)
summary["n_features"] = len(feature_cols)
summary["n_classes"] = n_classes
summary["n_splits"] = N_SPLITS
summary["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
summary["random_state"] = RANDOM_STATE

(REPORT_DIR / "results_xgb_v3_grouped.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

report = classification_report(
    oof_true,
    oof_pred,
    labels=np.arange(n_classes),
    target_names=class_names,
    zero_division=0
)

(REPORT_DIR / "classification_report_xgb_v3_grouped.txt").write_text(
    report,
    encoding="utf-8"
)

cm = confusion_matrix(
    oof_true,
    oof_pred,
    labels=np.arange(n_classes),
    normalize="true"
)

pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
).to_csv(REPORT_DIR / "confusion_matrix_xgb_v3_grouped.csv")


# ============================================================
# 6. REGISTRO EN MLFLOW
# ============================================================

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fallas-maquinaria")

    with mlflow.start_run(
        run_name="xgb_grouped_by_speed_v3"
    ):

        mlflow.log_params({
            "model": "XGBClassifier",
            "validation": "StratifiedGroupKFold",
            "group_variable": "nominal_speed_rpm",
            "inner_validation": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "n_features": len(feature_cols),
            "n_classes": n_classes,
            "max_depth": 4,
            "learning_rate": 0.05,
            "max_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "n_jobs": N_JOBS,
            "random_state": RANDOM_STATE,
            "xgboost_version": xgboost.__version__,
        })

        mlflow.log_metrics(summary)

        for file in [
            "cv_grouped_metrics_xgb_v3.csv",
            "oof_predictions_xgb_v3.csv",
            "results_xgb_v3_grouped.json",
            "classification_report_xgb_v3_grouped.txt",
            "confusion_matrix_xgb_v3_grouped.csv",
        ]:
            mlflow.log_artifact(str(REPORT_DIR / file))

        print("\nExperimento registrado en MLflow.")

else:
    print("\nMLFLOW_TRACKING_URI no configurada. Resultados guardados localmente.")


# ============================================================
# 7. RESUMEN FINAL
# ============================================================

print("\nRESULTADOS POR FOLD")
print(results_df.to_string(index=False))

print("\nRESULTADOS AGRUPADOS")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nREPORTE OOF")
print(report)

print("\nEVALUACIÓN AGRUPADA FINALIZADA.")
