"""
XGBoost V2 — diagnóstico de fallas MAFAULDA
Reproducibilidad, CV estratificada, early stopping interno, test reservado y MLflow.

Ejecución desde la raíz del repositorio:
    python src/train_xgboost.py

Variables opcionales:
    MLFLOW_TRACKING_URI  URI del servidor MLflow
    MAFAULDA_FEATURES    ruta al CSV de características
    XGB_N_JOBS           número de hilos (por defecto 1)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    learning_curve,
    validation_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv(
    "MAFAULDA_FEATURES", str(ROOT / "data" / "features_mafaulda.csv")
))
REPORT_DIR = ROOT / "reports" / "daniel"
MODEL_DIR = ROOT / "models" / "daniel"
MANIFEST_PATH = REPORT_DIR / "split_manifest_xgb_v1.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
INNER_VALIDATION_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 1500
N_JOBS = int(os.getenv("XGB_N_JOBS", "1"))

EXPERIMENT_NAME = "fallas-maquinaria"
RUN_NAME = "xgb_estratificado_v2_robusto"

# Lista explícita de canales y estadísticas admitidos. Evita incluir metadatos
# que casualmente compartan prefijos underhang_/overhang_.
CHANNELS = (
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
)
STATISTICS = (
    "mean", "std", "rms", "min", "max", "peak_to_peak",
    "peak", "skewness", "kurtosis", "crest_factor",
)
EXPECTED_FEATURES = {f"{channel}_{stat}" for channel in CHANNELS for stat in STATISTICS}
LEAKAGE_COLUMNS = {
    "fault_type", "fault_family", "bearing_position", "severity",
    "severity_unit", "file", "filepath", "target_10",
    "nominal_frequency_hz", "nominal_speed_rpm",
}


# =============================================================================
# 2. UTILIDADES Y VALIDACIONES
# =============================================================================

def sha256_file(path: Path) -> str:
    """Hash del archivo fuente para identificar exactamente el dataset."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_target(df: pd.DataFrame) -> pd.Series:
    """Construye las diez clases sin utilizar la etiqueta como predictor."""
    required = {"fault_type", "bearing_position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para construir target_10: {sorted(missing)}")

    target = df["fault_type"].astype(str).copy()
    bearing_mask = df["fault_type"].isin(["ball_fault", "cage_fault", "outer_race"])

    if df.loc[bearing_mask, "bearing_position"].isna().any():
        raise ValueError("Hay fallas de rodamiento sin bearing_position.")

    target.loc[bearing_mask] = (
        df.loc[bearing_mask, "bearing_position"].astype(str)
        + "_"
        + df.loc[bearing_mask, "fault_type"].astype(str)
    )
    return target.rename("target_10")


def validate_and_select_features(df: pd.DataFrame) -> list[str]:
    """Valida el esquema esperado, no solo el número de columnas."""
    available = set(df.columns)
    missing = EXPECTED_FEATURES - available
    if missing:
        raise ValueError(
            "El dataset no cumple el esquema de características V1. "
            f"Faltan: {sorted(missing)}"
        )

    selected = sorted(EXPECTED_FEATURES)
    forbidden = set(selected) & LEAKAGE_COLUMNS
    if forbidden:
        raise ValueError(f"Se detectaron columnas de metadata entre predictors: {forbidden}")

    non_numeric = [c for c in selected if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Características no numéricas: {non_numeric}")

    all_null = [c for c in selected if df[c].isna().all()]
    if all_null:
        raise ValueError(f"Características completamente nulas: {all_null}")

    return selected


def load_or_create_manifest(
    df: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Reutiliza el manifiesto V1 si existe. Si no existe, crea el split estratificado.
    El formato se conserva exactamente: filepath, target_10, partition.
    """
    if "filepath" not in df.columns:
        raise ValueError("El dataset no contiene la columna filepath.")
    if df["filepath"].isna().any():
        raise ValueError("filepath contiene valores nulos.")
    if not df["filepath"].is_unique:
        raise ValueError("filepath debe ser único: no se puede construir un manifiesto inequívoco.")

    current = pd.DataFrame({
        "filepath": df["filepath"].astype(str),
        "target_10": y.astype(str),
    })

    if MANIFEST_PATH.exists():
        old = pd.read_csv(MANIFEST_PATH, dtype={"filepath": str, "target_10": str})
        required = ["filepath", "target_10", "partition"]
        if list(old.columns) != required:
            raise ValueError(
                f"El manifiesto existente debe conservar las columnas {required}; "
                f"se encontraron {list(old.columns)}."
            )
        if old["filepath"].isna().any() or old["filepath"].duplicated().any():
            raise ValueError("El manifiesto existente tiene filepath nulos o duplicados.")
        if set(old["partition"].unique()) != {"train", "test"}:
            raise ValueError("El manifiesto debe contener únicamente train y test.")

        merged = current.merge(
            old, on=["filepath", "target_10"], how="left",
            validate="one_to_one", indicator=True,
        )
        if (merged["_merge"] != "both").any() or merged["partition"].isna().any():
            raise ValueError(
                "El manifiesto V1 no coincide con el dataset actual (filepath/target). "
                "No se sobrescribirá automáticamente."
            )
        manifest = merged[required].copy()
    else:
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        partition = np.full(len(df), "train", dtype=object)
        partition[test_idx] = "test"
        manifest = current.copy()
        manifest["partition"] = partition
        manifest = manifest[["filepath", "target_10", "partition"]]
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST_PATH, index=False)

    train_idx = np.flatnonzero(manifest["partition"].to_numpy() == "train")
    test_idx = np.flatnonzero(manifest["partition"].to_numpy() == "test")
    if set(train_idx) & set(test_idx) or len(train_idx) + len(test_idx) != len(df):
        raise ValueError("Partición inválida en el manifiesto.")

    return manifest, train_idx, test_idx


def log_class_balance(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Devuelve y registra conteos/proporciones por partición."""
    balance = pd.concat(
        [
            y_train.value_counts().rename("train_count"),
            y_test.value_counts().rename("test_count"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    balance["train_fraction"] = balance["train_count"] / len(y_train)
    balance["test_fraction"] = balance["test_count"] / len(y_test)
    balance.index.name = "class"
    balance.to_csv(REPORT_DIR / "class_balance_xgb_v2.csv")
    return balance


def new_classifier(n_classes: int, n_estimators: int = MAX_ESTIMATORS,
                   max_depth: int = 4, early_stopping: bool = True) -> XGBClassifier:
    kwargs = dict(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
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
    if early_stopping:
        kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**kwargs)


# =============================================================================
# 3. VALIDACIÓN CRUZADA CON EARLY STOPPING INTERNO
# =============================================================================

def run_cross_validation(X: pd.DataFrame, y: np.ndarray, class_names: list[str]):
    """
    En cada fold:
      - fold de validación externo: solo evaluación;
      - split interno del fold-train: early stopping;
      - imputador ajustado solo con el subtrain interno.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_iterations = []

    for fold, (outer_train, outer_valid) in enumerate(cv.split(X, y), start=1):
        X_outer_train, X_outer_valid = X.iloc[outer_train], X.iloc[outer_valid]
        y_outer_train, y_outer_valid = y[outer_train], y[outer_valid]

        inner_train, inner_valid = train_test_split(
            np.arange(len(outer_train)),
            test_size=INNER_VALIDATION_SIZE,
            random_state=RANDOM_STATE + fold,
            stratify=y_outer_train,
        )
        X_inner_train = X_outer_train.iloc[inner_train]
        X_inner_valid = X_outer_train.iloc[inner_valid]
        y_inner_train = y_outer_train[inner_train]
        y_inner_valid = y_outer_train[inner_valid]

        imputer = SimpleImputer(strategy="median")
        X_inner_train_imp = imputer.fit_transform(X_inner_train)
        X_inner_valid_imp = imputer.transform(X_inner_valid)
        X_outer_valid_imp = imputer.transform(X_outer_valid)

        model = new_classifier(len(class_names))
        model.fit(
            X_inner_train_imp, y_inner_train,
            eval_set=[(X_inner_valid_imp, y_inner_valid)],
            verbose=False,
        )

        # Evaluación externa: no fue usada para ajustar ni detener el modelo.
        pred_valid = model.predict(X_outer_valid_imp)
        pred_train = model.predict(X_inner_train_imp)

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y_outer_valid, pred_valid),
            "f1_macro": f1_score(y_outer_valid, pred_valid, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_outer_valid, pred_valid),
            "train_accuracy": accuracy_score(y_inner_train, pred_train),
            "train_f1_macro": f1_score(y_inner_train, pred_train, average="macro", zero_division=0),
            "train_balanced_accuracy": balanced_accuracy_score(y_inner_train, pred_train),
            "best_iteration": int(model.best_iteration),
        })
        best_iterations.append(int(model.best_iteration) + 1)

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"F1 macro={rows[-1]['f1_macro']:.4f}; "
            f"best_iteration={model.best_iteration}"
        )

    cv_results = pd.DataFrame(rows)
    cv_results.to_csv(REPORT_DIR / "cv_metrics_xgb_v2.csv", index=False)
    return cv_results, best_iterations


# =============================================================================
# 4. FUNCIONES DE GRÁFICAS (6)
# =============================================================================

def plot_learning_curve(X, y, estimator, cv, out_path: Path):
    """F1-macro train/validación según el tamaño de muestra."""
    sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="f1_macro",
        train_sizes=np.linspace(0.25, 1.0, 5),
        n_jobs=1, shuffle=True, random_state=RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(axis=1), marker="o", label="Train")
    ax.fill_between(
        sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15,
    )
    ax.plot(sizes, valid_scores.mean(axis=1), marker="o", label="CV")
    ax.fill_between(
        sizes, valid_scores.mean(axis=1) - valid_scores.std(axis=1),
        valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15,
    )
    ax.set(title="Learning curve — F1 macro", xlabel="Registros de entrenamiento",
           ylabel="F1 macro")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curve(X, y, n_estimators: int, cv, out_path: Path):
    """Compara complejidad max_depth con CV."""
    depths = [2, 3, 4, 5, 6]
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", new_classifier(
            n_classes=len(np.unique(y)), n_estimators=n_estimators,
            max_depth=4, early_stopping=False
        )),
    ])
    train_scores, valid_scores = validation_curve(
        estimator, X, y, param_name="classifier__max_depth",
        param_range=depths, cv=cv, scoring="f1_macro", n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_scores.mean(axis=1), marker="o", label="Train")
    ax.plot(depths, valid_scores.mean(axis=1), marker="o", label="CV")
    ax.fill_between(depths, valid_scores.mean(axis=1) - valid_scores.std(axis=1),
                    valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    ax.set(title="Validation curve — max_depth", xlabel="max_depth", ylabel="F1 macro")
    ax.set_xticks(depths)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_boosting_curve(eval_results: dict, best_iteration: int, out_path: Path):
    """Curvas mlogloss de train y validación interna."""
    train_loss = eval_results["validation_0"]["mlogloss"]
    valid_loss = eval_results["validation_1"]["mlogloss"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="Train interno")
    ax.plot(range(1, len(valid_loss) + 1), valid_loss, label="Validación interna")
    ax.axvline(best_iteration + 1, linestyle="--",
               label=f"best_iteration + 1 = {best_iteration + 1}")
    ax.set(title="Boosting curve — mlogloss", xlabel="Número de árboles",
           ylabel="Multiclass log loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_confusion(y_true, y_pred, labels, out_path: Path):
    """Matriz normalizada por fila: cada fila suma 1 (recall por clase)."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=ax)
    ax.set(title="Matriz de confusión normalizada por fila",
           xlabel="Predicción", ylabel="Clase real")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_class_recall_f1(y_true, y_pred, labels, out_path: Path):
    """Barras horizontales de recall y F1 por clase."""
    _, recall, f1_values, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    table = pd.DataFrame({"class": labels, "recall": recall, "f1": f1_values})
    table = table.sort_values("recall", ascending=True)
    table.to_csv(REPORT_DIR / "class_metrics_xgb_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(table))
    ax.barh(y_pos - 0.18, table["recall"], height=0.35, label="Recall")
    ax.barh(y_pos + 0.18, table["f1"], height=0.35, label="F1")
    ax.set_yticks(y_pos, table["class"])
    ax.set_xlim(0, 1.05)
    ax.set(title="Recall y F1 por clase — TEST", xlabel="Puntuación")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cv_boxplot(cv_results: pd.DataFrame, out_path: Path):
    """Distribución de métricas entre folds."""
    metrics = ["accuracy", "f1_macro", "balanced_accuracy"]
    long = cv_results.melt(
        id_vars="fold", value_vars=metrics,
        var_name="metric", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=long, x="metric", y="score", ax=ax)
    sns.stripplot(data=long, x="metric", y="score", color="black",
                  alpha=0.65, size=5, ax=ax)
    ax.set(title="Estabilidad entre folds", xlabel="", ylabel="Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. ENTRENAMIENTO FINAL, EVALUACIÓN Y MLflow
# =============================================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError(
            "Define MLFLOW_TRACKING_URI en la terminal antes de ejecutar el script."
        )

    print("Cargando dataset...")
    df = pd.read_csv(DATA_PATH)
    y_text = build_target(df)
    feature_cols = validate_and_select_features(df)
    X = df.loc[:, feature_cols].copy()

    if y_text.isna().any():
        raise ValueError("La variable objetivo contiene valores nulos.")
    class_counts = y_text.value_counts()
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"La clase minoritaria tiene {class_counts.min()} casos; "
            f"no alcanza para {N_SPLITS} folds estratificados."
        )

    manifest_was_reused = MANIFEST_PATH.exists()
    manifest, train_idx, test_idx = load_or_create_manifest(df, y_text)
    y_train_text = y_text.iloc[train_idx].reset_index(drop=True)
    y_test_text = y_text.iloc[test_idx].reset_index(drop=True)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    # Evitar características sin ningún dato válido en TRAIN
    empty_train_features = X_train.columns[X_train.isna().all()].tolist()

    if empty_train_features:
        raise ValueError(
            "Hay características completamente nulas en TRAIN: "
            f"{empty_train_features}"
        )

    # Encoder ajustado solo con las clases del TRAIN; TEST debe tener las mismas.
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    unseen = set(y_test_text.unique()) - set(encoder.classes_)
    if unseen:
        raise ValueError(f"Clases en TEST ausentes en TRAIN: {sorted(unseen)}")
    y_test = encoder.transform(y_test_text)
    class_names = encoder.classes_.tolist()

    balance = log_class_balance(y_train_text, y_test_text)
    print("\nBalance train/test:\n", balance.to_string())

    # La CV se ejecuta exclusivamente sobre el TRAIN.
    cv_results, best_iterations = run_cross_validation(
        X_train, y_train, class_names
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    selected_n_estimators = max(1, int(np.median(best_iterations)))

    # Split interno independiente para generar la curva de boosting.
    fit_idx, val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=INNER_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    curve_imputer = SimpleImputer(strategy="median")
    X_fit_imp = curve_imputer.fit_transform(X_train.iloc[fit_idx])
    X_val_imp = curve_imputer.transform(X_train.iloc[val_idx])
    curve_model = new_classifier(len(class_names))
    curve_model.fit(
        X_fit_imp, y_train[fit_idx],
        eval_set=[
            (X_fit_imp, y_train[fit_idx]),
            (X_val_imp, y_train[val_idx]),
        ],
        verbose=False,
    )
    boosting_best_iteration = int(curve_model.best_iteration)
    boosting_history = curve_model.evals_result()

    # Diagnósticos de curvas sobre TRAIN; no se consulta TEST.
    learning_estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", new_classifier(
            len(class_names), n_estimators=selected_n_estimators,
            early_stopping=False
        )),
    ])
    plot_learning_curve(
        X_train, y_train, learning_estimator, cv,
        REPORT_DIR / "learning_curve_xgb_v2.png"
    )
    plot_validation_curve(
        X_train, y_train, selected_n_estimators, cv,
        REPORT_DIR / "validation_curve_xgb_v2.png"
    )
    plot_boosting_curve(
        boosting_history, boosting_best_iteration,
        REPORT_DIR / "boosting_curve_xgb_v2.png"
    )
    plot_cv_boxplot(cv_results, REPORT_DIR / "cv_boxplot_xgb_v2.png")

    # Modelo final: imputador y XGBoost se ajustan con todo TRAIN.
    # n_estimators procede de la mediana de best_iteration de CV.
    final_model = new_classifier(
        len(class_names), n_estimators=selected_n_estimators,
        early_stopping=False,
    )
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", final_model),
    ])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }
    report_dict = classification_report(
        y_test, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0,
    )
    (REPORT_DIR / "classification_report_xgb_v2.txt").write_text(
        report_text, encoding="utf-8"
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        REPORT_DIR / "classification_report_xgb_v2.csv"
    )

    plot_normalized_confusion(
        y_test, y_pred, class_names,
        REPORT_DIR / "confusion_matrix_normalized_xgb_v2.png"
    )
    plot_class_recall_f1(
        y_test, y_pred, class_names,
        REPORT_DIR / "class_recall_f1_xgb_v2.png"
    )

    # Feature importance por gain (no confundir con causalidad).
    classifier = final_pipeline.named_steps["classifier"]
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": classifier.feature_importances_,
    }).sort_values("gain", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance_gain_xgb_v2.csv", index=False)

    # Serialización end-to-end: pipeline ya contiene el imputador; encoder aparte.
    bundle = {
        "pipeline": final_pipeline,
        "label_encoder": encoder,
        "feature_columns": feature_cols,
        "class_names": class_names,
        "target_definition": "fault_type + bearing_position for bearing faults",
    }
    model_path = MODEL_DIR / "xgboost_v2_bundle.joblib"
    joblib.dump(bundle, model_path)

    # Resultados CV: media/std y score de entrenamiento.
    cv_summary = {}
    for metric in ["accuracy", "f1_macro", "balanced_accuracy"]:
        cv_summary[f"cv_{metric}_mean"] = float(cv_results[metric].mean())
        cv_summary[f"cv_{metric}_std"] = float(cv_results[metric].std(ddof=1))
        train_metric = {
            "accuracy": "train_accuracy",
            "f1_macro": "train_f1_macro",
            "balanced_accuracy": "train_balanced_accuracy",
        }[metric]
        cv_summary[f"cv_{metric}_train_mean"] = float(cv_results[train_metric].mean())

    results = {
        **metrics,
        **cv_summary,
        "selected_n_estimators_median_cv": selected_n_estimators,
        "boosting_curve_best_iteration": boosting_best_iteration,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "random_state": RANDOM_STATE,
    }
    (REPORT_DIR / "results_xgb_v2.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Auditoría del entorno y de los datos.
    versions = {
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "mlflow_version": mlflow.__version__,
    }
    (REPORT_DIR / "environment_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )
    requirements_path = ROOT / "requirements.txt"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({
            "model": "XGBClassifier",
            "n_estimators_final": selected_n_estimators,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "n_jobs": N_JOBS,
            "random_state": RANDOM_STATE,
            "cv_strategy": "StratifiedKFold(5) with inner early-stopping split",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "test_size": TEST_SIZE,
            "inner_validation_size": INNER_VALIDATION_SIZE,
            "n_features": len(feature_cols),
            "n_classes": len(class_names),
        })
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_metrics({k: float(v) for k, v in cv_summary.items()})
        mlflow.log_metric("selected_n_estimators_median_cv", selected_n_estimators)
        mlflow.log_metric("boosting_curve_best_iteration", boosting_best_iteration)
        mlflow.log_param("data_sha256", sha256_file(DATA_PATH))
        mlflow.log_param("data_file", str(DATA_PATH.name))
        mlflow.log_param("manifest_file", str(MANIFEST_PATH.name))
        mlflow.log_param("manifest_reused", str(manifest_was_reused))
        for key, value in versions.items():
            mlflow.log_param(key, value)

        mlflow.log_text(report_text, "classification_report_xgb_v2.txt")
        mlflow.log_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            "results_xgb_v2.json",
        )
        mlflow.log_text(
            json.dumps(versions, indent=2), "environment_versions.json"
        )

        artifact_files = [
            REPORT_DIR / "cv_metrics_xgb_v2.csv",
            REPORT_DIR / "class_balance_xgb_v2.csv",
            REPORT_DIR / "classification_report_xgb_v2.csv",
            REPORT_DIR / "classification_report_xgb_v2.txt",
            REPORT_DIR / "feature_importance_gain_xgb_v2.csv",
            REPORT_DIR / "class_metrics_xgb_v2.csv",
            REPORT_DIR / "results_xgb_v2.json",
            REPORT_DIR / "environment_versions.json",
            REPORT_DIR / "learning_curve_xgb_v2.png",
            REPORT_DIR / "validation_curve_xgb_v2.png",
            REPORT_DIR / "boosting_curve_xgb_v2.png",
            REPORT_DIR / "confusion_matrix_normalized_xgb_v2.png",
            REPORT_DIR / "class_recall_f1_xgb_v2.png",
            REPORT_DIR / "cv_boxplot_xgb_v2.png",
            model_path,
            MANIFEST_PATH,
        ]
        for artifact in artifact_files:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        if requirements_path.exists():
            mlflow.log_artifact(str(requirements_path))
        else:
            warnings.warn("No existe requirements.txt; no se pudo registrar.")

        # Modelo sklearn empaquetado con imputador. Encoder se guarda en bundle.
        mlflow.sklearn.log_model(final_pipeline, name="xgboost_pipeline")
        mlflow.log_dict(
            {"classes": class_names, "features": feature_cols},
            "inference_schema.json",
        )

        print("\nMÉTRICAS TEST:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("\nCV (media ± std):")
        for metric in ["accuracy", "f1_macro", "balanced_accuracy"]:
            print(
                f"{metric}: {cv_results[metric].mean():.4f} "
                f"± {cv_results[metric].std(ddof=1):.4f}"
            )
        print("\nREPORTE DE CLASIFICACIÓN:\n", report_text)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"MLflow: {tracking_uri}/#/experiments/1/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
