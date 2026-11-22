"""
Prepara el archivo de lecturas que usa el tablero (dashboard/data/lecturas.csv).

Toma la tabla de 78 características (data/features_mafaulda_v2.csv) y el
manifiesto de partición, y conserva solo lo que el tablero necesita:
  - las 391 grabaciones de PRUEBA, que el usuario puede diagnosticar;
  - las grabaciones NORMALES de ENTRENAMIENTO, que sirven como referencia
    en la comparación "lectura frente a condición normal".

Las 78 columnas de características se toman del modelo V4, para que los nombres
coincidan exactamente con los que espera la API.

Cómo correrlo (desde la raíz del repositorio, con el ambiente activo):
    python dashboard/preparar_lecturas.py
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLA = ROOT / "data" / "features_mafaulda_v2.csv"
MANIFIESTO = ROOT / "reports" / "daniel" / "split_manifest_xgb_v1.csv"
MODELO = ROOT / "models" / "daniel" / "xgboost_v4_bundle.joblib"
SALIDA = ROOT / "dashboard" / "data" / "lecturas.csv"


def clave_archivo(ruta):
    """Parte de la ruta desde MAFAULDA, para que coincida en cualquier computador."""
    ruta = str(ruta).replace("\\", "/")
    return ruta.split("MAFAULDA/")[-1].lower()


def main():
    caracteristicas = joblib.load(MODELO)["feature_cols"]

    tabla = pd.read_csv(TABLA)
    manifiesto = pd.read_csv(MANIFIESTO)
    tabla["clave"] = tabla["filepath"].map(clave_archivo)
    manifiesto["clave"] = manifiesto["filepath"].map(clave_archivo)

    datos = tabla.merge(
        manifiesto[["clave", "target_10", "partition"]],
        on="clave", how="inner", validate="one_to_one",
    )
    if len(datos) != len(manifiesto):
        raise ValueError("No todas las grabaciones del manifiesto se encontraron en la tabla.")

    es_prueba = datos["partition"] == "test"
    es_normal_entrenamiento = (datos["partition"] == "train") & (datos["target_10"] == "normal")
    datos = datos[es_prueba | es_normal_entrenamiento]

    metadatos = ["clave", "nominal_frequency_hz", "nominal_speed_rpm", "target_10", "partition"]
    salida = datos[metadatos + list(caracteristicas)].rename(
        columns={"target_10": "clase_real", "partition": "particion"}
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA, index=False)
    print(f"Archivo generado: {SALIDA}")
    print(f"  Lecturas de prueba: {(salida['particion'] == 'test').sum()}")
    print(f"  Normales de referencia (entrenamiento): {(salida['particion'] == 'train').sum()}")
    print(f"  Características por lectura: {len(caracteristicas)}")


if __name__ == "__main__":
    main()
