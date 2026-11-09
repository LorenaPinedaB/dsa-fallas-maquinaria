# Diagnóstico automático de fallas en maquinaria rotativa

Proyecto del curso **Despliegue de Soluciones Analíticas (MIID-4304)**, Maestría en Inteligencia Analítica de Datos (MIAD), Universidad de los Andes. **Grupo 15:** Daniel Serna, Brian Ospina y Lorena Pineda.

El proyecto clasifica, a partir de una grabación de vibración de 5 segundos, si una máquina rotativa opera en condición normal o presenta una de nueve fallas (desbalanceo, desalineación horizontal o vertical, y defectos de rodamiento según tipo y posición). La solución incluye modelos supervisados, seguimiento de experimentos en MLflow y un tablero en Dash. La API de inferencia y el despliegue con Docker se agregarán en la entrega final.

## Estructura del repositorio

```
├── dashboard/
│   └── app.py                      Tablero de diagnóstico (Dash)
├── data/                           Datos gestionados con DVC (no se suben a GitHub)
│   ├── features_mafaulda.csv.dvc   Tabla de características temporales (1.951 × 69)
│   └── raw/MAFAULDA.dvc            Dataset original
├── models/
│   └── xgboost_v2_bundle.joblib.dvc  Modelo usado por el tablero (versionado con DVC)
├── notebooks/
│   └── eda_brian.ipynb             Análisis exploratorio de datos
├── reports/
│   └── daniel/                     Resultados de los experimentos XGBoost y manifiesto de partición
├── src/
│   ├── feature_extraction.py       Características temporales
│   ├── feature_extraction_v2.py    Características espectrales (1×, 2× y 3× la frecuencia de giro)
│   ├── train.py                    Script base: regresión logística y Random Forest
│   └── train_xgboost*.py           Experimentos XGBoost V1 a V4
└── requirements.txt
```

## Datos

Se usa la base pública **MaFaulDa** (Machinery Fault Database) del laboratorio SMT de la Universidade Federal do Rio de Janeiro: http://www02.smt.ufrj.br/~offshore/mfs/page_01.html

Los datos y el modelo se versionan con DVC en un bucket de Amazon S3. Descargarlos con `dvc pull` requiere credenciales de AWS con acceso al bucket del proyecto. El dataset original también se puede descargar directamente de la fuente.

## Cómo ejecutar el tablero

1. Crear y activar un ambiente virtual (Python 3.11 o superior) e instalar las dependencias:
   ```bash
   python -m venv env-proyecto
   source env-proyecto/Scripts/activate    # En Linux o Mac: source env-proyecto/bin/activate
   pip install -r requirements.txt
   ```
2. Obtener los datos y el modelo:
   ```bash
   dvc pull data/features_mafaulda.csv.dvc models/xgboost_v2_bundle.joblib.dvc
   ```
3. Desde la raíz del repositorio, ejecutar:
   ```bash
   python dashboard/app.py
   ```
4. Abrir http://127.0.0.1:8050 en el navegador.

## Experimentos

Los experimentos se registran en un servidor de MLflow desplegado en AWS EC2, en el experimento `fallas-maquinaria`. Para registrar nuevas corridas, se define la dirección del servidor antes de ejecutar los scripts:

```bash
export MLFLOW_TRACKING_URI=http://<IP-DEL-SERVIDOR>:5000
```

## Flujo de trabajo del equipo

Cada integrante trabaja en una rama propia y los cambios se integran a `main` mediante Pull Requests revisados. No se suben datos ni credenciales a Git.
