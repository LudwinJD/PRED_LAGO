import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('DATA LAGO.csv', delimiter=';', encoding='latin1')

# Preprocesamiento
df['FECHA_HORA'] = pd.to_datetime(df['FECHA'] + ' ' + df['HORA'], format='%d/%m/%Y %H:%M')
df['AÑO'] = df['FECHA_HORA'].dt.year
df['MES'] = df['FECHA_HORA'].dt.month
df['DIA'] = df['FECHA_HORA'].dt.day
df['HORA'] = df['FECHA_HORA'].dt.hour

# Convertir columnas a numéricas
columnas_numericas = [
    'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 
    'HUMEDAD (%)', 'DIRECCION DEL VIENTO (°)', 
    'VELOCIDAD DEL VIENTO (m/s)', 'NIVEL_LAGO'
]
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Preparar características y target
X = df[['AÑO', 'MES', 'DIA', 'HORA', 
        'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 
        'HUMEDAD (%)', 'DIRECCION DEL VIENTO (°)', 
        'VELOCIDAD DEL VIENTO (m/s)']]
y = df['NIVEL_LAGO']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo Random Forest
modelo = RandomForestRegressor(n_estimators=200, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Predicciones
y_pred = modelo.predict(X_test_scaled)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio: {mse}")
print(f"R2 Score: {r2}")

# Función de predicción
def predecir_nivel_lago(año, mes, dia, hora, temp, precip, humedad, dir_viento, vel_viento):
    # Preparar datos de entrada
    entrada = np.array([[año, mes, dia, hora, temp, precip, humedad, dir_viento, vel_viento]])
    entrada_escalada = scaler.transform(entrada)
    prediccion = modelo.predict(entrada_escalada)
    return prediccion[0]

# Ejemplo de predicción
ejemplo = predecir_nivel_lago(2023, 11, 28, 10, 12.8, 0, 33, 110, 2.3)
print(f"Nivel de lago predicho: {ejemplo}")

# Gráfico de predicciones vs reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Nivel de Lago Real")
plt.ylabel("Nivel de Lago Predicho")
plt.title("Predicciones vs Valores Reales")
plt.show()