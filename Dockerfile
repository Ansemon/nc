FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema para netCDF
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Puerto para Dash
EXPOSE 8050

# Comando para ejecutar la app
CMD ["python", "app.py"]
