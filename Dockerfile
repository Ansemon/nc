FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación y datos
COPY app.py .
COPY Land_and_Ocean_LatLong1.nc .

# Exponer puerto y ejecutar
EXPOSE 8050
CMD ["python", "app.py"]