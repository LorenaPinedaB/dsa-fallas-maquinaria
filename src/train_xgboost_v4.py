
import os
import json
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
# 1. CONFIGURACION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features_mafaulda_v2.csv"
MANIFEST_PATH = ROOT / "reports" / "daniel" / "split_manifest_xgb_v1.csv"
REPORT_DIR = ROOT / "reports" / "daniel"

RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 1500
N_JOBS = 1

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGA Y PREPARACION
# ============================================================

print("Cargando dataset espectral...")

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

# Verificar integridad del manifiesto
if df["filepath"].duplicated().any():
    raise ValueError("Hay filepath duplicados en el dataset.")

if manifest["filepath"].duplicated().any():
    raise ValueError("Hay filepath duplicados en el manifiesto.")

if set(df["filepath"]) != set(manifest["filepath"]):
    raise ValueError("Dataset y manifiesto no contienen los mismos archivos.")

# Recuperar TRAIN y TEST oficiales
train_manifest = manifest.loc[
    manifest["partition"] == "train",
    ["filepath", "target_10"]
]

test_manifest = manifest.loc[
    manifest["partition"] == "test",
    ["filepath", "target_10"]
]

df_train = df.merge(
    train_manifest,
    on="filepath",
    how="inner",
    suffixes=("", "_manifest"),
    validate="one_to_one"
)

df_test = df.merge(
    test_manifest,
    on="filepath",
    how="inner",
    suffixes=("", "_manifest"),
    validate="one_to_one"
)

for name, data, expected in [
    ("TRAIN", df_train, len(train_manifest)),
    ("TEST", df_test, len(test_manifest)),
]:
    if len(data) != expected:
        raise ValueError(f"No coinciden los registros de {name}.")

    if not (data["target_10"] == data["target_10_manifest"]).all():
        raise ValueError(f"Los targets de {name} no coinciden con el manifiesto.")

    data.drop(columns=["target_10_manifest"], inplace=True)

# Seleccionar explicitamente las 60 caracteristicas temporales
temporal_features = [
    c for c in df.columns
    if any(
        c == f"{channel}_{stat}"
        for channel in [
            "underhang_axial",
            "underhang_radial",
            "underhang_tangential",
            "overhang_axial",
            "overhang_radial",
            "overhang_tangential",
        ]
        for stat in [
            "mean", "std", "rms", "min", "max",
            "peak_to_peak", "peak", "skewness",
            "kurtosis", "crest_factor",
        ]
    )
]

# Seleccionar las 18 caracteristicas espectrales
spectral_features = [
    c for c in df.columns
    if c.endswith("_order_1x_amplitude")
    or c.endswith("_order_2x_amplitude")
    or c.endswith("_order_3x_amplitude")
]

feature_cols = sorted(temporal_features + spectral_features)

if len(temporal_features) != 60:
    raise ValueError(
        f"Se esperaban 60 caracteristicas temporales: {len(temporal_features)}"
    )

if len(spectral_features) != 18:
    raise ValueError(
        f"Se esperaban 18 caracteristicas espectrales: {len(spectral_features)}"
    )

if len(feature_cols) != 78:
    raise ValueError(
        f"Se esperaban 78 caracteristicas totales: {len(feature_cols)}"
    )

X = df_train[feature_cols].copy()
y_text = df_train["target_10"].reset_index(drop=True)
groups = df_train["nominal_speed_rpm"].reset_index(drop=True)

X_test = df_test[feature_cols].copy()
y_test_text = df_test["target_10"].reset_index(drop=True)

if groups.isna().any():
    raise ValueError("Hay velocidades nominales nulas.")

if X.isna().all().any() or X_test.isna().all().any():
    raise ValueError("Hay caracteristicas completamente nulas.")

if X.isna().sum().sum() or X_test.isna().sum().sum():
    print("Aviso: existen valores nulos. Se imputaran usando TRAIN.")

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

# Verificar que TEST tenga las mismas clases
unknown = set(y_test_text) - set(encoder.classes_)

if unknown:
    raise ValueError(f"Clases desconocidas en TEST: {unknown}")

y_test = encoder.transform(y_test_text)

class_names = encoder.classes_.tolist()
n_classes = len(class_names)

print("TRAIN:", len(X))
print("TEST:", len(X_test))
print("Caracteristicas temporales:", len(temporal_features))
print("Caracteristicas espectrales:", len(spectral_features))
print("Total caracteristicas:", len(feature_cols))
print("Clases:", n_classes)
print("Velocidades:", groups.nunique())


# ============================================================
# 3. MODELO
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
# 4. VALIDACION CRUZADA AGRUPADA
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

print("\nINICIANDO VALIDACION AGRUPADA POR VELOCIDAD - V4")

for fold, (outer_train, outer_valid) in enumerate(
    outer_cv.split(X, y, groups), start=1
):

    X_outer_train = X.iloc[outer_train]
    X_outer_valid = X.iloc[outer_valid]

    y_outer_train = y[outer_train]
    y_outer_valid = y[outer_valid]

    groups_outer_train = groups.iloc[outer_train]
    groups_outer_valid = groups.iloc[outer_valid]

    overlap = set(groups_outer_train).intersection(
        set(groups_outer_valid)
    )

    if overlap:
        raise ValueError(f"Solapamiento de velocidades en fold {fold}")

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
        raise ValueError("Solapamiento en validacion interna.")

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

    pred = model.predict(X_outer_valid_imp)

    accuracy = accuracy_score(y_outer_valid, pred)
    f1 = f1_score(
        y_outer_valid,
        pred,
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
# 5. RESULTADOS CV
# ============================================================

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    REPORT_DIR / "cv_grouped_metrics_xgb_v4.csv",
    index=False
)

oof_df = pd.DataFrame(oof_records)

oof_df.to_csv(
    REPORT_DIR / "oof_predictions_xgb_v4.csv",
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

summary["group_oof_accuracy"] = accuracy_score(oof_true, oof_pred)

summary["group_oof_f1_macro"] = f1_score(
    oof_true,
    oof_pred,
    average="macro",
    zero_division=0
)

summary["group_oof_balanced_accuracy"] = balanced_accuracy_score(
    oof_true,
    oof_pred
)

summary["n_records_train"] = len(X)
summary["n_features_temporal"] = len(temporal_features)
summary["n_features_spectral"] = len(spectral_features)
summary["n_features_total"] = len(feature_cols)
summary["n_classes"] = n_classes
summary["n_splits"] = N_SPLITS
summary["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
summary["random_state"] = RANDOM_STATE

report = classification_report(
    oof_true,
    oof_pred,
    labels=np.arange(n_classes),
    target_names=class_names,
    zero_division=0
)

(REPORT_DIR / "classification_report_xgb_v4_grouped.txt").write_text(
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
).to_csv(
    REPORT_DIR / "confusion_matrix_xgb_v4_grouped.csv"
)


# ============================================================
# 6. MODELO FINAL Y TEST OFICIAL
# ============================================================

print("\nENTRENANDO MODELO FINAL CON TRAIN OFICIAL...")

# Seleccion de estimadores basada en los folds agrupados
final_estimators = max(
    1,
    int(np.median(results_df["best_iteration"])) + 1
)

final_imputer = SimpleImputer(strategy="median")

X_train_imp = final_imputer.fit_transform(X)
X_test_imp = final_imputer.transform(X_test)

final_model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    n_estimators=final_estimators,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    importance_type="gain",
)

final_model.fit(
    X_train_imp,
    y
)
# ============================================================
# GUARDAR MODELO FINAL V4
# ============================================================

import joblib

MODEL_DIR = ROOT / "models" / "daniel"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

v4_bundle = {
    "model": final_model,
    "imputer": final_imputer,
    "feature_cols": feature_cols,
    "class_names": class_names,
    "version": "XGBoost_V4",
    "n_features": len(feature_cols),
}

joblib.dump(
    v4_bundle,
    MODEL_DIR / "xgboost_v4_bundle.joblib"
)

print("Modelo V4 guardado correctamente.")

test_pred = final_model.predict(X_test_imp)

test_metrics = {
    "test_accuracy": accuracy_score(y_test, test_pred),
    "test_f1_macro": f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    ),
    "test_balanced_accuracy": balanced_accuracy_score(
        y_test,
        test_pred
    ),
    "final_n_estimators": final_estimators,
}

summary.update(test_metrics)

test_report = classification_report(
    y_test,
    test_pred,
    labels=np.arange(n_classes),
    target_names=class_names,
    zero_division=0
)

(REPORT_DIR / "classification_report_xgb_v4_test.txt").write_text(
    test_report,
    encoding="utf-8"
)

test_cm = confusion_matrix(
    y_test,
    test_pred,
    labels=np.arange(n_classes),
    normalize="true"
)

pd.DataFrame(
    test_cm,
    index=class_names,
    columns=class_names
).to_csv(
    REPORT_DIR / "confusion_matrix_xgb_v4_test.csv"
)

pd.DataFrame({
    "filepath": df_test["filepath"].values,
    "target_true": y_test_text.values,
    "target_pred": encoder.inverse_transform(test_pred),
}).to_csv(
    REPORT_DIR / "test_predictions_xgb_v4.csv",
    index=False
)

(REPORT_DIR / "results_xgb_v4.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


# ============================================================
# 7. IMPORTANCIA DE CARACTERISTICAS
# ============================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_gain": final_model.feature_importances_,
})

importance = importance.sort_values(
    "importance_gain",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "feature_importance_xgb_v4.csv",
    index=False
)


# ============================================================
# 8. REGISTRO MLFLOW
# ============================================================

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fallas-maquinaria")

    with mlflow.start_run(run_name="xgb_spectral_orders_v4"):

        mlflow.log_params({
            "model": "XGBClassifier",
            "version": "V4_spectral_orders",
            "validation": "StratifiedGroupKFold",
            "group_variable": "nominal_speed_rpm",
            "inner_validation": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "n_features_temporal": len(temporal_features),
            "n_features_spectral": len(spectral_features),
            "n_features_total": len(feature_cols),
            "n_classes": n_classes,
            "max_depth": 4,
            "learning_rate": 0.05,
            "max_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "final_n_estimators": final_estimators,
            "n_jobs": N_JOBS,
            "random_state": RANDOM_STATE,
            "frequency_tachometer_records": int(
                (df["frequency_source"] == "tachometer").sum()
            ),
            "frequency_nominal_fallback_records": int(
                (df["frequency_source"] == "nominal_fallback").sum()
            ),
            "xgboost_version": xgboost.__version__,
        })

        mlflow.log_metrics(summary)

        for filename in [
            "cv_grouped_metrics_xgb_v4.csv",
            "oof_predictions_xgb_v4.csv",
            "classification_report_xgb_v4_grouped.txt",
            "confusion_matrix_xgb_v4_grouped.csv",
            "classification_report_xgb_v4_test.txt",
            "confusion_matrix_xgb_v4_test.csv",
            "test_predictions_xgb_v4.csv",
            "feature_importance_xgb_v4.csv",
            "results_xgb_v4.json",
        ]:
            mlflow.log_artifact(str(REPORT_DIR / filename))

        print("\nExperimento V4 registrado en MLflow.")

else:
    print("\nMLFLOW_TRACKING_URI no configurada.")


# ============================================================
# 9. RESUMEN FINAL
# ============================================================

print("\nRESULTADOS POR FOLD")
print(results_df.to_string(index=False))

print("\nRESULTADOS AGRUPADOS")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nREPORTE OOF")
print(report)

print("\nRESULTADOS TEST OFICIAL")
print(test_metrics)

print("\nREPORTE TEST")
print(test_report)

print("\nEXPERIMENTO XGBOOST V4 FINALIZADO.")
