# Diagnóstico automático de fallas en maquinaria rotativa

Proyecto del curso **Despliegue de Soluciones Analíticas (MIID-4304)**, Maestría en Inteligencia Analítica de Datos (MIAD), Universidad de los Andes. **Grupo 15:** Daniel Serna, Brian Ospina y Lorena Pineda.

El proyecto clasifica, a partir de una grabación de vibración de 5 segundos, si una máquina rotativa opera en condición normal o presenta una de nueve fallas (desbalanceo, desalineación horizontal o vertical, y defectos de rodamiento según tipo y posición). La solución completa incluye modelos supervisados homologados, seguimiento de experimentos en MLflow, una API de inferencia en FastAPI, un tablero interactivo en Dash y despliegue empaquetado mediante contenedores Docker.


## Estructura del repositorio

```text
├── api/
│   ├── app/
│   │   ├── main.py                         API de inferencia (FastAPI)
│   │   └── models/
│   │       └── xgboost_v4_bundle.joblib    Modelo empaquetado de producción (XGBoost V4, 78 features)
│   ├── tests/                              Pruebas unitarias y payloads de validación
│   ├── Dockerfile                          Imagen Docker de la API
│   └── requirements.txt                    Dependencias de la API
├── dashboard/
│   ├── data/
│   │   └── lecturas.csv                    Lecturas de prueba para el tablero
│   ├── app.py                              Tablero de diagnóstico (Dash)
│   ├── preparar_lecturas.py                Script de preparación de lecturas de prueba
│   ├── Dockerfile                          Imagen Docker del tablero
│   └── requirements.txt                    Dependencias del tablero
├── data/                                   Datos gestionados con DVC (no se suben a GitHub)
│   ├── features_mafaulda.csv.dvc           Tabla de características temporales (1.951 × 69)
│   ├── features_mafaulda_v2.csv.dvc        Tabla maestra temporal + espectral (1.951 × 87, 78 features)
│   └── raw/MAFAULDA.dvc                    Dataset original
├── docs/img/manual/                        Capturas usadas en el manual de usuario
├── notebooks/
│   └── eda_brian.ipynb                     Análisis exploratorio de datos
├── reports/
│   ├── brian/                              Métricas, reportes y matrices de confusión homologadas (LR y RF)
│   └── daniel/                             Resultados de experimentos XGBoost V1–V4 y manifiesto de partición
├── src/
│   ├── feature_extraction.py               Extracción de características temporales (60 features)
│   ├── feature_extraction_v2.py            Extracción de características espectrales (1×, 2× y 3× giro)
│   ├── train.py                            Entrenamiento homologado de modelos de referencia (LR y RF)
│   └── train_xgboost*.py                   Experimentos XGBoost V1 a V4
├── docker-compose.yml                      Orquestación de contenedores (API + Tablero)
├── MANUAL_INSTALACION.md                   Manual detallado de instalación y despliegue
├── MANUAL_USUARIO.md                       Manual de uso e interpretación del tablero
└── requirements.txt                        Dependencias completas del entorno de entrenamiento
```


## Datos

Se usa la base pública **MaFaulDa** (Machinery Fault Database) del laboratorio SMT de la Universidade Federal do Rio de Janeiro: http://www02.smt.ufrj.br/~offshore/mfs/page_01.html

Los datos crudos y las tablas de características (`features_mafaulda.csv` y `features_mafaulda_v2.csv`) se versionan con DVC en Amazon S3. Descargarlos con `dvc pull` requiere credenciales de AWS con acceso al bucket del proyecto. Para facilitar la ejecución inmediata del tablero y la API, el modelo final (`xgboost_v4_bundle.joblib`) y las lecturas de prueba (`lecturas.csv`) ya se encuentran incluidos en sus respectivas carpetas.



## Orden de Ejecución del Pipeline (Extracción a Entrenamiento)

El flujo de datos y modelado del proyecto sigue una secuencia determinística en cinco etapas:

1. **Ingesta de Señales Crudas (`data/raw/MAFAULDA/`)**
   * Lectura de los 1.951 registros de vibración (5 segundos a 50 kHz, 8 canales: tacómetro, 6 acelerómetros en *underhang* y *overhang*, y micrófono), versionados mediante DVC.
2. **Extracción de Características Temporales (`src/feature_extraction.py`)**
   * Cálculo de 10 descriptores estadísticos en el dominio del tiempo (media, desviación estándar, RMS, mínimo, máximo, amplitud pico a pico, pico, asimetría, curtosis y factor de cresta) sobre los 6 canales de vibración, generando 60 variables predictoras.
3. **Extracción de Características Espectrales (`src/feature_extraction_v2.py`)**
   * Estimación de la frecuencia de giro a partir del tacómetro (con respaldo de frecuencia nominal en 19 registros) y cálculo de las amplitudes espectrales en los órdenes 1X, 2X y 3X para cada canal, consolidando la tabla maestra de **78 características** (`data/features_mafaulda_v2.csv`).
4. **Partición Determinística y Preprocesamiento (`src/train.py`)**
   * Aplicación del manifiesto fijo compartido (`reports/daniel/split_manifest_xgb_v1.csv`: `1.560` observaciones de entrenamiento y `391` de prueba) sobre la variable objetivo `target_10` (10 clases).
   * Exclusión de metadatos y de la velocidad de giro (`nominal_speed_rpm`) para evitar fugas de información.
   * Imputación por mediana y estandarización (`StandardScaler`) ajustadas exclusivamente sobre el conjunto de entrenamiento (`X_train`).
5. **Entrenamiento Homologado y Registro en MLflow (`src/train.py` y `src/train_xgboost_v4.py`)**
   * Ejecución comparativa de **Regresión Logística** (línea base), **Random Forest** y **XGBoost V4** bajo idénticas condiciones experimentales, registrando parámetros, métricas (`accuracy`, `f1_macro`, `balanced_accuracy`, `recall_normal`), matrices de confusión y artefactos serializados en el servidor MLflow sobre AWS EC2.



## Instalación y Ejecución de la Solución

Para instrucciones detalladas paso a paso tanto con Docker Compose como en ejecución local con entornos virtuales Python, consulte el **[Manual de Instalación](MANUAL_INSTALACION.md)**.

### Inicio rápido con Docker Compose
```bash
docker compose up --build -d
```
* **Tablero (Dash):** http://localhost:8050
* **API (FastAPI / Swagger):** http://localhost:8000/docs



## Experimentos en MLflow

Los experimentos se registran en un servidor de MLflow desplegado en AWS EC2, en el experimento `fallas-maquinaria`. Para registrar nuevas corridas, se define la dirección del servidor antes de ejecutar los scripts:

```bash
# En Linux / Mac / Git Bash:
export MLFLOW_TRACKING_URI=http://<IP-DEL-SERVIDOR>:5000

# En Windows PowerShell:
$env:MLFLOW_TRACKING_URI="http://<IP-DEL-SERVIDOR>:5000"
```


## Flujo de trabajo del equipo

Cada integrante trabaja en una rama propia y los cambios se integran a `main` mediante Pull Requests revisados. No se suben datos crudos ni credenciales a Git.