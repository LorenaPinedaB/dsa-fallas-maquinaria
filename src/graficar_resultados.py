"""
Generación de gráficas para el informe DSA - Grupo 15
Autor: Daniel
Ejecutar desde la raíz del repositorio:
    python src/graficar_resultados.py

No entrena modelos. Lee los resultados ya generados en reports/daniel/
y guarda las figuras en reports/daniel/figuras_adicionales/.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "daniel"
OUT = REPORTS / "figuras_adicionales"
OUT.mkdir(parents=True, exist_ok=True)

# Estilo sobrio para informe académico: se usan colores predeterminados de Matplotlib.
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
})


def find_column(df, candidates):
    """Busca una columna ignorando mayúsculas, espacios y guiones."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    # búsqueda parcial
    for key, original in normalized.items():
        if any(re.sub(r"[^a-z0-9]", "", c.lower()) in key for c in candidates):
            return original
    return None


def save_fig(fig, filename):
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path.relative_to(ROOT)}")


def read_csv(filename, **kwargs):
    path = REPORTS / filename
    if not path.exists():
        print(f"[OMITIDA] No existe: {path}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[OMITIDA] No se pudo leer {filename}: {exc}")
        return None


def plot_class_balance():
    df = read_csv("class_balance_xgb_v2.csv")
    if df is None or df.empty:
        return

    class_col = find_column(df, ["class", "target", "target_10", "label", "clase"])
    count_col = find_column(df, ["count", "counts", "n", "support", "total", "cantidad"])

    if class_col is None or count_col is None:
        print("[OMITIDA] class_balance_xgb_v2.csv: no identifiqué columnas de clase y frecuencia.")
        print("Columnas encontradas:", list(df.columns))
        return

    df = df.sort_values(count_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df[class_col].astype(str), df[count_col])
    ax.set_title("Distribución de observaciones por clase")
    ax.set_xlabel("Número de registros")
    ax.set_ylabel("Clase")
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, df[count_col]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f" {int(value)}", va="center", fontsize=8)
    save_fig(fig, "Distribución de clases en el conjunto de entrenamiento.png")


def plot_confusion_matrix():
    df = read_csv("confusion_matrix_xgb_v4_test.csv", index_col=0)
    if df is None or df.empty:
        return

    # El CSV puede tener la primera columna como etiquetas y las restantes como clases.
    try:
        matrix = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    except Exception:
        print("[OMITIDA] No fue posible convertir la matriz de confusión a valores numéricos.")
        return

    labels = [str(x) for x in df.index]
    if len(labels) != matrix.shape[0]:
        labels = [str(i) for i in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, label="Número de observaciones")
    ax.set(
        title="Matriz de confusión — XGBoost V4 (conjunto de prueba)",
        xlabel="Clase predicha",
        ylabel="Clase real",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
    )
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_yticklabels(labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > threshold else "black"
            )
    save_fig(fig, "02_matriz_confusion_v4_test.png")


def parse_classification_report():
    path = REPORTS / "classification_report_xgb_v4_test.txt"
    if not path.exists():
        print(f"[OMITIDA] No existe: {path}")
        return None

    rows = []
    # Formato habitual de sklearn: clase precision recall f1-score support
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            label, precision, recall, f1, support = match.groups()
            rows.append({
                "class": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            })

    if not rows:
        print("[OMITIDA] No pude interpretar el reporte de clasificación V4.")
        print("Revisa el formato de classification_report_xgb_v4_test.txt.")
        return None

    return pd.DataFrame(rows)


def plot_class_metrics():
    df = parse_classification_report()
    if df is None:
        return

    x = np.arange(len(df))
    width = .24
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, df["precision"], width, label="Precision")
    ax.bar(x, df["recall"], width, label="Recall")
    ax.bar(x + width, df["f1"], width, label="F1-score")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Métrica")
    ax.set_title("Desempeño por clase — XGBoost V4 (prueba)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=3, frameon=False)
    save_fig(fig, "03_metricas_por_clase_v4.png")

    # Figura complementaria: soporte de cada clase en prueba
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df["class"], df["support"])
    ax.set_title("Número de observaciones por clase en el conjunto de prueba")
    ax.set_ylabel("Soporte")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["class"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=.25)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    save_fig(fig, "04_soporte_por_clase_test.png")


def plot_v3_v4_comparison():
    rows = [
        ("Accuracy", 96.9215, 97.5018),
        ("F1 macro", 96.4626, 97.0350),
        ("Balanced accuracy", 96.0107, 96.5886),
    ]
    labels = [r[0] for r in rows]
    v3 = [r[1] for r in rows]
    v4 = [r[2] for r in rows]
    x = np.arange(len(labels))
    width = .34

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - width/2, v3, width, label="V3: 60 características temporales")
    b2 = ax.bar(x + width/2, v4, width, label="V4: 60 temporales + 18 espectrales")
    ax.set_ylim(90, 100)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Comparación de V3 y V4 — validación agrupada por velocidad")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.12,
                    f"{bar.get_height():.2f}%", ha="center", fontsize=8)
    save_fig(fig, "05_comparacion_v3_v4.png")


def plot_cv_folds():
    df = read_csv("cv_grouped_metrics_xgb_v4.csv")
    if df is None or df.empty:
        return

    fold_col = find_column(df, ["fold", "split", "fold_id"])
    metric_cols = {
        "Accuracy": find_column(df, ["accuracy", "acc"]),
        "F1 macro": find_column(df, ["f1_macro", "f1macro", "f1"]),
        "Balanced accuracy": find_column(df, ["balanced_accuracy", "balancedacc"]),
    }
    metric_cols = {k: v for k, v in metric_cols.items() if v is not None}
    if not metric_cols:
        print("[OMITIDA] No identifiqué métricas en cv_grouped_metrics_xgb_v4.csv.")
        print("Columnas encontradas:", list(df.columns))
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    for label, col in metric_cols.items():
        ax.plot(x, pd.to_numeric(df[col], errors="coerce")*100,
                marker="o", linewidth=1.8, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df[fold_col].astype(str) if fold_col else [f"Fold {i+1}" for i in range(len(df))]
    )
    ax.set_xlabel("Partición de validación")
    ax.set_ylabel("Métrica (%)")
    ax.set_title("Variabilidad del desempeño entre folds — XGBoost V4")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    save_fig(fig, "06_metricas_por_fold_v4.png")


def plot_feature_importance():
    df = read_csv("feature_importance_xgb_v4.csv")
    if df is None or df.empty:
        return

    feature_col = find_column(df, ["feature", "feature_name", "variable", "name"])
    importance_col = find_column(df, ["gain", "importance", "weight", "score", "total_gain"])
    if feature_col is None or importance_col is None:
        print("[OMITIDA] No identifiqué las columnas de variable e importancia.")
        print("Columnas encontradas:", list(df.columns))
        return

    df[importance_col] = pd.to_numeric(df[importance_col], errors="coerce")
    df = df.dropna(subset=[importance_col]).nlargest(20, importance_col).sort_values(importance_col)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df[feature_col].astype(str), df[importance_col])
    ax.set_title("Veinte características con mayor importancia — XGBoost V4")
    ax.set_xlabel(f"Importancia ({importance_col})")
    ax.set_ylabel("Característica")
    ax.grid(axis="x", alpha=.25)
    save_fig(fig, "07_importancia_variables_v4.png")


def main():
    print(f"Directorio de resultados: {REPORTS}")
    print(f"Las figuras se guardarán en: {OUT}")
    plot_class_balance()
    plot_confusion_matrix()
    plot_class_metrics()
    plot_v3_v4_comparison()
    plot_cv_folds()
    plot_feature_importance()
    print("\nProceso terminado. Revisa las figuras generadas y las advertencias anteriores.")


if __name__ == "__main__":
    main()
