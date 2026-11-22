
FROM python:3.12-slim

# Crear usuario de ejecución
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/maquinaria-api

# Instalar dependencias
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación y modelo V4
COPY api/app ./app

# Permisos de ejecución
RUN chown -R api-user:api-user /opt/maquinaria-api

USER api-user

# Puerto utilizado por la plataforma
EXPOSE 8000

# Iniciar FastAPI
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
