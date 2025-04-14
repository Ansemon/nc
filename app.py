import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import base64
import io
from statsmodels.tsa.stattools import acf, pacf
import os
import urllib.request
# Cargar el dataset

# Ruta local y URL del archivo
local_nc_file = "Land_and_Ocean_LatLong1.nc"
dropbox_url = "https://www.dropbox.com/scl/fi/pps7v2vdtydptcad97uqf/Land_and_Ocean_LatLong1.nc?rlkey=dwqdti8wpwoztjbk3yrdqkvau&dl=1"

# Descargar el archivo si no existe localmente
if not os.path.exists(local_nc_file):
    print("Descargando archivo NetCDF desde Dropbox...")
    urllib.request.urlretrieve(dropbox_url, local_nc_file)
    print("Descarga completada.")
dataset = xr.open_dataset(local_nc_file, engine='netcdf4')
# Preprocesamiento de datos (similar al notebook)
dataset_interp_time = dataset["temperature"].interp(time=dataset["time"].values, method="linear")
dataset_interp_space = dataset_interp_time.interpolate_na(dim="latitude", method="nearest")
dataset_interp_space = dataset_interp_space.interpolate_na(dim="longitude", method="nearest")

# Calcular promedio global anual de temperatura
years = dataset_interp_space["time"].values.astype(int)
global_avg = dataset_interp_space.mean(dim=["latitude", "longitude"])
# Calcular ACF y PACF
acf_vals = acf(global_avg, nlags=20)
pacf_vals = pacf(global_avg, nlags=20, method='yw')
# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1("Análisis de Temperatura Global (1850-2025)", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        # Pestaña 1: Resumen del Dataset
        dcc.Tab(label='Resumen del Dataset', children=[
            html.Div([
                html.H3("Información General del Dataset"),
                html.P("""
                    Este dataset proporciona información detallada sobre la temperatura de la superficie terrestre 
                    desde el año 1850 hasta 2025. Contiene datos organizados en dimensiones espaciales y temporales, 
                    incluyendo longitud (en grados este), latitud (en grados norte), tiempo (en años) y un identificador 
                    numérico de los meses del año.
                """),
                html.H4("Variables Principales:"),
                html.Ul([
                    html.Li("land_mask: Distingue entre regiones terrestres y oceánicas"),
                    html.Li("temperature: Anomalía de temperatura de la superficie terrestre"),
                    html.Li("climatology: Promedios de temperatura de 1951 a 1980")
                ]),
                html.H3("Estadísticas Descriptivas de Temperatura"),
                html.P("""
                    - Cantidad de datos: 120,281,700 valores registrados
                    - Promedio: 0.0745°C
                    - Desviación estándar: 1.2636°C
                    - Mínimo: -19.39°C, Máximo: 22.23°C
                """),
            ], style={'padding': '20px'})
        ]),
        
        # Pestaña 2: Datos Faltantes
        dcc.Tab(label='Datos Faltantes', children=[
            html.Div([
                html.H3("Análisis de Datos Faltantes"),
                html.P("""
                    Se identificaron 15.8 millones de datos faltantes en la variable 'temperature', 
                    principalmente concentrados en el siglo XIX. La interpolación redujo este número a 12.3 millones.
                """),
                dcc.Graph(
                    id='missing-data-evolution',
                    figure={
                        'data': [
                            go.Scatter(
                                x=years,
                                y=dataset["temperature"].isnull().sum(dim=["latitude", "longitude"]),
                                mode='lines+markers',
                                name='Datos Faltantes'
                            )
                        ],
                        'layout': go.Layout(
                            title='Evolución de Datos Faltantes en "temperature"',
                            xaxis={'title': 'Año'},
                            yaxis={'title': 'Cantidad de Datos Faltantes'}
                        )
                    }
                ),
                html.H4("Mapa de Datos Faltantes Antes y Después de Interpolación"),
                dcc.Graph(
                    id='missing-data-maps',
                    figure={
                        'data': [
                            go.Heatmap(
                                x=dataset["longitude"].values,
                                y=dataset["latitude"].values,
                                z=dataset["temperature"].isnull().sum(dim="time"),
                                colorscale='Reds',
                                name='Antes'
                            ),
                            go.Heatmap(
                                x=dataset["longitude"].values,
                                y=dataset["latitude"].values,
                                z=dataset_interp_space.isnull().sum(dim="time"),
                                colorscale='Reds',
                                name='Después',
                                visible=False
                            )
                        ],
                        'layout': go.Layout(
                            title='Datos Faltantes por Ubicación Geográfica',
                            updatemenus=[{
                                'buttons': [
                                    {'label': 'Antes de Interpolación', 'method': 'update', 'args': [{'visible': [True, False]}]},
                                    {'label': 'Después de Interpolación', 'method': 'update', 'args': [{'visible': [False, True]}]}
                                ],
                                'direction': 'down',
                                'showactive': True,
                            }]
                        )
                    }
                ),
                html.H4("Evaluación de la Interpolación"),
                html.P("""
                    - Error Medio Absoluto (MAE): 0.0022
                    - Raíz del Error Cuadrático Medio (RMSE): 0.0524
                    La interpolación preservó la distribución original sin introducir distorsiones significativas.
                """)
            ], style={'padding': '20px'})
        ]),
        
        # Pestaña 3: Tendencia Global
# Pestaña 3: Tendencia Global (Versión Corregida)
        dcc.Tab(label='Tendencia Global', children=[
            html.Div([
                html.H3("Tendencia Global de Temperatura"),
                dcc.Graph(
                    id='global-trend',
                    figure={
                        'data': [
                            go.Scatter(
                                x=years,
                                y=global_avg,
                                mode='lines',
                                name='Anomalía de Temperatura'
                            ),
                            go.Scatter(
                                x=years,
                                y=np.poly1d(np.polyfit(years, global_avg, 1))(years),
                                mode='lines',
                                name=f'Tendencia: {np.polyfit(years, global_avg, 1)[0]*100:.2f}°C/siglo',
                                line={'dash': 'dash'}
                            )
                        ],
                        'layout': go.Layout(
                            title='Tendencia Global de Temperatura (1850-2025)',
                            xaxis={'title': 'Año'},
                            yaxis={'title': 'Anomalía de Temperatura (°C)'}
                        )
                    }
                ),
                
                html.H3("Descomposición de la Serie Temporal"),
                dcc.Graph(
                    id='time-series-decomposition',
                    figure=px.line(
                        seasonal_decompose(
                            pd.Series(
                                global_avg, 
                                index=pd.to_datetime(years, format='%Y')
                            ),  # Cierre correcto de pd.Series
                            model='additive', 
                            period=10
                        ).observed.reset_index(),  # Cierre correcto de seasonal_decompose
                        x='index', 
                        y=0,
                        title='Componente Observada'
                    )
                ),
                
                html.H3("Prueba de Estacionariedad"),
                html.P(f"""
                    - Estadístico ADF: {adfuller(global_avg)[0]} 
                    - p-value: {adfuller(global_avg)[1]}
                    - La serie NO es estacionaria
                """)
            ], style={'padding': '20px'})
        ]),
        
        # Pestaña 4: Autocorrelación
        dcc.Tab(label='Autocorrelación', children=[
            html.Div([
                html.H3("Análisis de Autocorrelación"),
                html.P("""
                    Los gráficos ACF y PACF revelan la estructura temporal de la serie:
                """),
                # Gráfico ACF
                dcc.Graph(
                    id='acf-plot',
                    figure={
                        'data': [
                            go.Scatter(
                                x=list(range(len(acf_vals))),
                                y=acf_vals,
                                mode='lines+markers',
                                name='ACF'
                            )
                        ],
                        'layout': go.Layout(
                            title='Función de Autocorrelación (ACF)',
                            xaxis={'title': 'Lag'},
                            yaxis={'title': 'Autocorrelación'}
                        )
                    }
                ),
                # Gráfico PACF
                dcc.Graph(
                    id='pacf-plot',
                    figure={
                        'data': [
                            go.Scatter(
                                x=list(range(len(pacf_vals))),
                                y=pacf_vals,
                                mode='lines+markers',
                                name='PACF'
                            )
                        ],
                        'layout': go.Layout(
                            title='Función de Autocorrelación Parcial (PACF)',
                            xaxis={'title': 'Lag'},
                            yaxis={'title': 'Autocorrelación Parcial'}
                        )
                    }
                ),
                html.H4("Interpretación:"),
                html.Ul([
                    html.Li("ACF: Decaimiento lento indica tendencia persistente (no estacionariedad)"),
                    html.Li("PACF: Corte abrupto en lag 1 sugiere un modelo AR(1) es apropiado")
                ])
            ], style={'padding': '20px'})
        ])
    ])
])

# Ejecutar la aplicación

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Puerto dinámico para Railway
    app.run(debug=False, host='0.0.0.0', port=port)