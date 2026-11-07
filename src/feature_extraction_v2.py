
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "MAFAULDA"

OUTPUT_FILE = PROJECT_ROOT / "data" / "features_mafaulda_v2.csv"

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


def extract_metadata(filepath):
    parts = filepath.parts
    idx = parts.index("MAFAULDA")
    relative_parts = parts[idx + 1:]

    filename = filepath.stem

    try:
        nominal_frequency = float(filename)
    except ValueError:
        nominal_frequency = np.nan

    metadata = {
        "file": filepath.name,
        "filepath": str(filepath),
        "nominal_frequency_hz": nominal_frequency,
        "nominal_speed_rpm": nominal_frequency * 60,
    }

    family = relative_parts[0]
    metadata["fault_family"] = family

    if family == "normal":
        metadata.update({
            "fault_type": "normal",
            "severity": 0,
            "severity_unit": None,
            "bearing_position": None,
        })

    elif family == "imbalance":
        metadata.update({
            "fault_type": "imbalance",
            "severity": float(relative_parts[1].replace("g", "")),
            "severity_unit": "g",
            "bearing_position": None,
        })

    elif family in ["horizontal-misalignment",
                    "vertical-misalignment"]:
        metadata.update({
            "fault_type": family.replace("-", "_"),
            "severity": float(relative_parts[1].replace("mm", "")),
            "severity_unit": "mm",
            "bearing_position": None,
        })

    elif family in ["underhang", "overhang"]:
        metadata.update({
            "fault_type": relative_parts[1],
            "severity": float(relative_parts[2].replace("g", "")),
            "severity_unit": "g",
            "bearing_position": family,
        })

    else:
        metadata.update({
            "fault_type": "unknown",
            "severity": np.nan,
            "severity_unit": None,
            "bearing_position": None,
        })

    return metadata


def extract_signal_features(signal):
    x = np.asarray(signal, dtype=np.float64)

    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))

    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "rms": rms,
        "min": np.min(x),
        "max": np.max(x),
        "peak_to_peak": np.ptp(x),
        "peak": peak,
        "skewness": skew(x),
        "kurtosis": kurtosis(x),
        "crest_factor": peak / rms if rms != 0 else np.nan,
    }


#def estimate_rotation_frequency(tachometer, nominal_frequency):
#    """
#    Estima la frecuencia de giro mediante los flancos ascendentes
#    del tacómetro. Utiliza la frecuencia nominal únicamente como
#    referencia para descartar intervalos incompatibles.
#    """
#
#    tach = np.asarray(tachometer, dtype=np.float64)
#
#    threshold = (np.min(tach) + np.max(tach)) / 2
#    digital = tach > threshold
#
#    rising_edges = np.where(
#        np.diff(digital.astype(np.int8)) == 1
#    )[0]
#
#    if len(rising_edges) < 3:
#        raise ValueError("No hay suficientes pulsos de tacómetro")
#
#    intervals = np.diff(rising_edges)
#
#    expected_period = SAMPLING_RATE / nominal_frequency
#
#    # El tacómetro puede presentar pulsos adicionales.
#    # Se conservan intervalos compatibles con la velocidad esperada.
#    valid = intervals[
#        (intervals >= expected_period * 0.95) &
#        (intervals <= expected_period * 1.05)
#    ]
#
#    if len(valid) < 3:
#        raise ValueError(
#            "No se pudo estimar una velocidad confiable desde el tacómetro"
#        )
#
#    measured_frequency = SAMPLING_RATE / np.median(valid)
#
#    return measured_frequency


def estimate_rotation_frequency(tachometer, nominal_frequency):

    tach = np.asarray(tachometer, dtype=np.float64)

    threshold = (np.min(tach) + np.max(tach)) / 2
    digital = tach > threshold

    rising_edges = np.where(
        np.diff(digital.astype(np.int8)) == 1
    )[0]

    if len(rising_edges) < 3:
        return nominal_frequency, "nominal_fallback"

    intervals = np.diff(rising_edges)
    expected_period = SAMPLING_RATE / nominal_frequency

    valid = intervals[
        (intervals >= expected_period * 0.95) &
        (intervals <= expected_period * 1.05)
    ]

    if len(valid) < 3:
        return nominal_frequency, "nominal_fallback"

    measured_frequency = SAMPLING_RATE / np.median(valid)

    return measured_frequency, "tachometer"


def extract_order_features(signal, rotation_frequency):
    """
    Amplitud espectral alrededor de 1X, 2X y 3X.
    Se utiliza interpolación entre bins de la FFT.
    """

    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)

    n = len(x)

    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * window))

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / SAMPLING_RATE
    )

    # Corrección de amplitud de la ventana.
    spectrum = 2 * spectrum / np.sum(window)

    features = {}

    for order in [1, 2, 3]:

        target_frequency = order * rotation_frequency

        amplitude = np.interp(
            target_frequency,
            frequencies,
            spectrum
        )

        features[f"order_{order}x_amplitude"] = amplitude

    return features


def process_file(filepath):

    df = pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS
    )

    row = extract_metadata(filepath)

    nominal_frequency = row["nominal_frequency_hz"]

    if not np.isfinite(nominal_frequency) or nominal_frequency <= 0:
        raise ValueError("Frecuencia nominal inválida")

    # measured_frequency = estimate_rotation_frequency(
    #     df["tachometer"],
    #     nominal_frequency
    # )

    # row["measured_rotation_hz"] = measured_frequency
    # row["measured_speed_rpm"] = measured_frequency * 60

    measured_frequency, frequency_source = estimate_rotation_frequency(
    df["tachometer"],
    nominal_frequency
)

    row["measured_rotation_hz"] = measured_frequency
    row["measured_speed_rpm"] = measured_frequency * 60
    row["frequency_source"] = frequency_source

    for channel in VIBRATION_CHANNELS:

        signal = df[channel].to_numpy(dtype=np.float64)

        # Las 60 características temporales originales.
        for name, value in extract_signal_features(signal).items():
            row[f"{channel}_{name}"] = value

        # Las tres características espectrales nuevas.
        for name, value in extract_order_features(
            signal,
            measured_frequency
        ).items():
            row[f"{channel}_{name}"] = value

    return row


def main():

    print("Buscando archivos MaFaulDa...")

    files = sorted(DATA_DIR.rglob("*.csv"))

    print(f"Archivos encontrados: {len(files)}")

    if not files:
        raise FileNotFoundError(DATA_DIR)

    rows = []
    errors = []

    for i, filepath in enumerate(files, start=1):

        try:
            rows.append(process_file(filepath))

        except Exception as error:
            errors.append((str(filepath), str(error)))

        if i % 100 == 0:
            print(f"Procesados: {i}/{len(files)}")

    result = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== RESULTADOS ==========")
    print(f"Registros procesados: {len(result)}")
    print(f"Errores: {len(errors)}")
    print(f"Columnas: {len(result.columns)}")
    print(f"Archivo: {OUTPUT_FILE}")

    if errors:
        error_file = PROJECT_ROOT / "reports" / "daniel" / "errors_feature_v2.csv"
        error_file.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            errors,
            columns=["filepath", "error"]
        ).to_csv(error_file, index=False)

        print(f"Detalle de errores: {error_file}")


if __name__ == "__main__":
    main()
