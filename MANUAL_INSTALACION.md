# Manual de Instalación y Despliegue — Diagnóstico Automático de Fallas

Este documento describe el procedimiento paso a paso para instalar, configurar y ejecutar desde cero la solución analítica (API de inferencia en FastAPI y Tablero interactivo en Dash), tanto mediante contenedores Docker como en entorno local.

## 1. Prerrequisitos del Sistema
* **Sistema Operativo:** Windows 10/11, macOS o Linux.
* **Git:** Instalado y disponible en la terminal.
* **Opción Contenedores:** Docker Desktop instalado y en ejecución (con soporte para Docker Compose).
* **Opción Ejecución Local:** Python 3.11 o superior instalado y agregado al `PATH`.

## 2. Clonación del Repositorio
Abra una terminal (PowerShell, Git Bash o terminal Unix) en el directorio donde desea instalar el proyecto y ejecute:

* **En Windows (PowerShell):**
  ```powershell
  git clone https://github.com/LorenaPinedaB/dsa-fallas-maquinaria.git
  cd dsa-fallas-maquinaria
  ```

* **En Git Bash / Linux / macOS:**
  ```bash
  git clone https://github.com/LorenaPinedaB/dsa-fallas-maquinaria.git
  cd dsa-fallas-maquinaria
  ```

> **Nota sobre artefactos:** El modelo empaquetado de producción (`api/app/models/xgboost_v4_bundle.joblib`) y el conjunto de lecturas de prueba del tablero (`dashboard/data/lecturas.csv`) ya se encuentran incluidos en el repositorio para permitir su despliegue inmediato sin pasos adicionales.

## 3. Despliegue mediante Docker Compose 
Este método construye y levanta automáticamente ambos servicios aislados y conectados en red (API en el puerto `8000` y Tablero en el puerto `8050`).

1. Desde la raíz del repositorio (`dsa-fallas-maquinaria`), ejecute:

* **En Windows (PowerShell):**
  ```powershell
  docker compose up --build -d
  ```
* **En Git Bash / Linux / macOS:**
  ```bash
  docker compose up --build -d
  ```

2. Verifique que ambos contenedores estén activos:

* **En Windows (PowerShell):**
  ```powershell
  docker compose ps
  ```

* **En Git Bash / Linux / macOS:**
  ```bash
  docker compose ps
  ```

3. **Acceso a los servicios:**
   * **Tablero Interactivo (Dash):** Abra en su navegador `http://localhost:8050`
   * **Documentación de la API (Swagger UI):** Abra en su navegador `http://localhost:8000/docs`

4. **Detener los servicios:** Cuando finalice el uso, detenga los contenedores ejecutando:

* **En Windows (PowerShell):**
  ```powershell
  docker compose down
  ```

* **En Git Bash / Linux / macOS:**
  ```bash
  docker compose down
  ```

## 4. Instalación y Ejecución Local (Sin Docker)
Si prefiere ejecutar los servicios directamente sobre Python local, debe abrir **dos terminales independientes** ubicadas en la raíz del proyecto (`dsa-fallas-maquinaria`).

### Terminal 1: Instalación y arranque de la API (FastAPI)
* **En Windows (PowerShell):**
  ```powershell
  python -m venv env-api
  .\env-api\Scripts\activate
  pip install --upgrade pip
  pip install -r api/requirements.txt
  uvicorn app.main:app --app-dir api --host 0.0.0.0 --port 8000
  ```
* **En Git Bash / Linux / macOS:**
  ```bash
  python -m venv env-api
  source env-api/Scripts/activate
  pip install --upgrade pip
  pip install -r api/requirements.txt
  uvicorn app.main:app --app-dir api --host 0.0.0.0 --port 8000
  ```

### Terminal 2: Instalación y arranque del Tablero (Dash)
* **En Windows (PowerShell):**
  ```powershell
  python -m venv env-dash
  .\env-dash\Scripts\activate
  pip install --upgrade pip
  pip install -r dashboard/requirements.txt
  $env:API_URL="http://localhost:8000"
  python dashboard/app.py
  ```
* **En Git Bash / Linux / macOS:**
  ```bash
  python -m venv env-dash
  source env-dash/Scripts/activate
  pip install --upgrade pip
  pip install -r dashboard/requirements.txt
  export API_URL=http://localhost:8000
  python dashboard/app.py
  ```

Una vez iniciados ambos servicios, ingrese a `http://127.0.0.1:8050` en su navegador para interactuar con el tablero conectado a la API.



## 5. Reproducción del Pipeline de Entrenamiento (Opcional)
Si desea reentrenar los modelos y registrar nuevos experimentos en MLflow:
1. Instale las dependencias completas de entrenamiento: `pip install -r requirements.txt`
2. Asegúrese de contar con la tabla de características `data/features_mafaulda_v2.csv` (gestionada mediante DVC).
3. Configure la URI del servidor MLflow y ejecute los scripts en orden:
   ```powershell
   $env:MLFLOW_TRACKING_URI=""
   python src/train.py
   python src/train_xgboost_v4.py
   ```