from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# CONFIGURACIÓN

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "MAFAULDA"

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "features_mafaulda.csv"

SAMPLING_RATE = 50_000

COLUMNS = [
    "tachometer",
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
    "microphone",
]

VIBRATION_CHANNELS = [
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
]


# METADATOS

def extract_metadata(filepath: Path) -> dict:

    parts = filepath.parts

    # Buscar la posición de MAFAULDA
    mafaulda_index = parts.index("MAFAULDA")

    relative_parts = parts[mafaulda_index + 1:]

    filename = filepath.stem

    metadata = {
        "file": filepath.name,
        "filepath": str(filepath),
    }

    # Velocidad nominal
    

    try:
        frequency_hz = float(filename)

        metadata["nominal_frequency_hz"] = frequency_hz
        metadata["nominal_speed_rpm"] = frequency_hz * 60

    except ValueError:

        metadata["nominal_frequency_hz"] = np.nan
        metadata["nominal_speed_rpm"] = np.nan

    
    # Clasificación según estructura de carpetas
    

    fault_family = relative_parts[0]

    metadata["fault_family"] = fault_family

    # NORMAL
    if fault_family == "normal":

        metadata["fault_type"] = "normal"
        metadata["severity"] = 0
        metadata["severity_unit"] = None
        metadata["bearing_position"] = None

    # IMBALANCE
    elif fault_family == "imbalance":

        severity_text = relative_parts[1]

        metadata["fault_type"] = "imbalance"
        metadata["severity"] = float(
            severity_text.replace("g", "")
        )
        metadata["severity_unit"] = "g"
        metadata["bearing_position"] = None

    # HORIZONTAL MISALIGNMENT
    elif fault_family == "horizontal-misalignment":

        severity_text = relative_parts[1]

        metadata["fault_type"] = "horizontal_misalignment"
        metadata["severity"] = float(
            severity_text.replace("mm", "")
        )
        metadata["severity_unit"] = "mm"
        metadata["bearing_position"] = None

    # VERTICAL MISALIGNMENT
    elif fault_family == "vertical-misalignment":

        severity_text = relative_parts[1]

        metadata["fault_type"] = "vertical_misalignment"
        metadata["severity"] = float(
            severity_text.replace("mm", "")
        )
        metadata["severity_unit"] = "mm"
        metadata["bearing_position"] = None

    # BEARING FAULTS
    elif fault_family in ["underhang", "overhang"]:

        metadata["bearing_position"] = fault_family

        metadata["fault_type"] = relative_parts[1]

        severity_text = relative_parts[2]

        metadata["severity"] = float(
            severity_text.replace("g", "")
        )

        metadata["severity_unit"] = "g"

    else:

        metadata["fault_type"] = "unknown"
        metadata["severity"] = np.nan
        metadata["severity_unit"] = None
        metadata["bearing_position"] = None

    return metadata


# CARACTERÍSTICAS DE UNA SEÑAL
 

def extract_signal_features(signal: pd.Series) -> dict:

    x = signal.to_numpy(dtype=np.float64)

    rms = np.sqrt(np.mean(x ** 2))

    peak = np.max(np.abs(x))

    features = {

        "mean": np.mean(x),

        "std": np.std(x),

        "rms": rms,

        "min": np.min(x),

        "max": np.max(x),

        "peak_to_peak": np.ptp(x),

        "peak": peak,

        "skewness": skew(x),

        "kurtosis": kurtosis(x),

        "crest_factor": (
            peak / rms
            if rms != 0
            else np.nan
        ),
    }

    return features


# PROCESAR UN CSV


def process_file(filepath: Path) -> dict:

    df = pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS
    )

    row = extract_metadata(filepath)

    # Características de vibración
    
    for channel in VIBRATION_CHANNELS:

        features = extract_signal_features(
            df[channel]
        )

        for feature_name, value in features.items():

            column_name = f"{channel}_{feature_name}"

            row[column_name] = value

    return row


# MAIN


def main():

    print("Buscando archivos CSV...")


    files = sorted(DATA_DIR.rglob("*.csv"))

#    files = [
#    DATA_DIR / "normal" / "12.288.csv",
 #   DATA_DIR / "horizontal-misalignment" / "0.5mm" / "12.288.csv",
  #  DATA_DIR / "vertical-misalignment" / "0.51mm" / "12.4928.csv",
   # DATA_DIR / "imbalance" / "6g" / "13.9264.csv",
    #DATA_DIR / "underhang" / "ball_fault" / "0g" / "13.1072.csv",
    #DATA_DIR / "overhang" / "ball_fault" / "0g" / "12.9024.csv",
#]

    print(f"Archivos encontrados: {len(files)}")

    if not files:

        raise FileNotFoundError(
            f"No se encontraron CSV en {DATA_DIR}"
        )

    rows = []

    for i, filepath in enumerate(files, start=1):

        print(
            f"[{i}/{len(files)}] {filepath.name}"
        )

        try:

            row = process_file(filepath)

            rows.append(row)

        except Exception as error:

            print(
                f"ERROR procesando {filepath}: {error}"
            )

    features_df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("===================================")
    print("Extracción terminada")
    print("===================================")
    print(f"Filas: {features_df.shape[0]}")
    print(f"Columnas: {features_df.shape[1]}")
    print(f"Archivo: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()