# Imagen del tablero de diagnóstico (Dash).
# Se construye desde la carpeta dashboard/:  docker build -t tablero-fallas ./dashboard
FROM python:3.12-slim

# Usuario sin privilegios para ejecutar la aplicación
RUN adduser --disabled-password --gecos '' tablero-user

WORKDIR /opt/tablero

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aplicación y lecturas de prueba
COPY app.py .
COPY data ./data

RUN chown -R tablero-user:tablero-user /opt/tablero
USER tablero-user

# Dirección de la API (se puede cambiar al ejecutar el contenedor con -e API_URL=...)
ENV API_URL=http://localhost:8000 \
    HOST=0.0.0.0 \
    PORT=8050

EXPOSE 8050

CMD ["python", "app.py"]
