
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATA_PATH = "data/features_mafaulda.csv"
MANIFEST_PATH = "reports/daniel/split_manifest_xgb_v1.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# 2. CARGA Y CONSTRUCCIÓN DE TARGET
# ============================================================

df = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

df["target_10"] = df["fault_type"].astype(str)

bearing_faults = ["ball_fault", "cage_fault", "outer_race"]
mask = df["fault_type"].isin(bearing_faults)

df.loc[mask, "target_10"] = (
    df.loc[mask, "bearing_position"].astype(str)
    + "_"
    + df.loc[mask, "fault_type"].astype(str)
)

# ============================================================
# 3. RECUPERAR ÚNICAMENTE EL TRAIN OFICIAL
# ============================================================

train_manifest = manifest.loc[
    manifest["partition"] == "train",
    ["filepath", "target_10"]
].copy()

if train_manifest["filepath"].isna().any():
    raise ValueError("Hay filepath nulos en el manifiesto.")

if train_manifest["filepath"].duplicated().any():
    raise ValueError("Hay filepath duplicados en el manifiesto.")

df_train = df.merge(
    train_manifest,
    on="filepath",
    how="inner",
    suffixes=("", "_manifest"),
    validate="one_to_one"
)

if len(df_train) != len(train_manifest):
    raise ValueError("No todos los registros del manifiesto fueron recuperados.")

if not (
    df_train["target_10"] == df_train["target_10_manifest"]
).all():
    raise ValueError("Los targets no coinciden con el manifiesto.")

df_train = df_train.drop(columns=["target_10_manifest"])

X = df_train.filter(regex=r"^(underhang|overhang)_")
y = df_train["target_10"]
groups = df_train["nominal_speed_rpm"]

if groups.isna().any():
    raise ValueError("Hay velocidades nulas.")

classes = sorted(y.unique())

print("\nREGISTROS TRAIN:", len(df_train))
print("FEATURES:", X.shape[1])
print("CLASES:", len(classes))
print("VELOCIDADES ÚNICAS:", groups.nunique())

print("\nDISTRIBUCIÓN ORIGINAL:")
print(y.value_counts().sort_index().to_string())

# ============================================================
# 4. STRATIFIED GROUP K-FOLD
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
all_valid_groups = []

print("\n" + "=" * 65)
print("VALIDACIÓN CRUZADA AGRUPADA POR VELOCIDAD")
print("=" * 65)

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    groups_train = set(groups.iloc[train_idx])
    groups_valid = set(groups.iloc[valid_idx])

    overlap = groups_train.intersection(groups_valid)

    missing_train = set(classes) - set(y_train.unique())
    missing_valid = set(classes) - set(y_valid.unique())

    print(f"\nFOLD {fold}")
    print("-" * 45)
    print("Train:", len(train_idx))
    print("Validación:", len(valid_idx))
    print("Velocidades train:", len(groups_train))
    print("Velocidades validación:", len(groups_valid))
    print("Velocidades compartidas:", len(overlap))
    print("Clases ausentes en train:", sorted(missing_train))
    print("Clases ausentes en validación:", sorted(missing_valid))

    print("\nDistribución de validación:")
    print(
        y_valid.value_counts()
        .reindex(classes, fill_value=0)
        .to_string()
    )

    if overlap:
        raise ValueError("Hay velocidades compartidas entre train y validación.")

    all_valid_groups.extend(groups_valid)

    results.append({
        "fold": fold,
        "train_records": len(train_idx),
        "valid_records": len(valid_idx),
        "train_groups": len(groups_train),
        "valid_groups": len(groups_valid),
        "missing_train_classes": len(missing_train),
        "missing_valid_classes": len(missing_valid),
        "group_overlap": len(overlap)
    })

# ============================================================
# 5. RESUMEN
# ============================================================

summary = pd.DataFrame(results)

print("\n" + "=" * 65)
print("RESUMEN")
print("=" * 65)
print(summary.to_string(index=False))

print("\n¿Todas las clases aparecen en todos los folds?")
print(
    summary["missing_valid_classes"].eq(0).all()
)

print("\n¿No existe solapamiento de velocidades?")
print(
    summary["group_overlap"].eq(0).all()
)

print("\n¿Todas las velocidades fueron evaluadas?")
print(
    len(set(all_valid_groups)) == groups.nunique()
)
