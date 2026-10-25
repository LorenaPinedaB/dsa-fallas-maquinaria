import pandas as pd

ruta = r"F:\PROYECTO_DESPLIEGUE\datos\MAFAULDA\normal\12.288.csv"

columnas = [
    "tachometer",
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
    "microphone",
]

df = pd.read_csv(ruta, header=None, names=columnas)

print("SHAPE:")
print(df.shape)

print("\nCOLUMNAS:")
print(df.columns.tolist())

print("\nPRIMERAS FILAS:")
print(df.head())

print("\nESTADISTICAS:")
print(df.describe().T)