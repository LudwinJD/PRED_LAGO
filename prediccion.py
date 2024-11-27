import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Cargar los datos desde un archivo CSV especificando el delimitador y la codificación correcta
df = pd.read_csv('DATA LAGO.csv', delimiter=';', encoding='latin1')

# Limpiar los nombres de las columnas (remueve espacios en blanco)
df.columns = df.columns.str.strip()

# Crear la columna 'FECHA_HORA' combinando 'FECHA' y 'HORA'
df['FECHA_HORA'] = pd.to_datetime(df['FECHA'] + ' ' + df['HORA'], format='%d/%m/%Y %H:%M')

# Crear nuevas columnas a partir de la fecha: año, día, mes y hora
df['AÑO'] = df['FECHA_HORA'].dt.year
df['DIA'] = df['FECHA_HORA'].dt.day
df['MES'] = df['FECHA_HORA'].dt.month
df['HORA'] = df['FECHA_HORA'].dt.hour

# Verificar que las columnas se han creado correctamente
print("Columnas disponibles en el DataFrame:", df.columns)

# Eliminar las columnas 'FECHA_HORA', 'FECHA' que ya no se usarán
df.drop(['FECHA_HORA', 'FECHA'], axis=1, inplace=True)

# Verificar los tipos de datos y convertir las columnas a numéricas si es necesario
df['TEMPERATURA (°C)'] = pd.to_numeric(df['TEMPERATURA (°C)'], errors='coerce')
df['PRECIPITACION (mm/hora)'] = pd.to_numeric(df['PRECIPITACION (mm/hora)'], errors='coerce')
df['HUMEDAD (%)'] = pd.to_numeric(df['HUMEDAD (%)'], errors='coerce')
df['DIRECCION DEL VIENTO (°)'] = pd.to_numeric(df['DIRECCION DEL VIENTO (°)'], errors='coerce')
df['VELOCIDAD DEL VIENTO (m/s)'] = pd.to_numeric(df['VELOCIDAD DEL VIENTO (m/s)'], errors='coerce')
df['NIVEL_LAGO'] = pd.to_numeric(df['NIVEL_LAGO'], errors='coerce')

# Eliminar cualquier fila con valores faltantes
df.dropna(inplace=True)

# Definir las características (X) y la variable objetivo (y)
X = df[['AÑO', 'DIA', 'MES', 'HORA', 'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 'HUMEDAD (%)', 'DIRECCION DEL VIENTO (°)', 'VELOCIDAD DEL VIENTO (m/s)']]
y = df['NIVEL_LAGO']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
error_cuadratico_medio = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {error_cuadratico_medio}")

# Función para predecir el nivel del lago para una nueva fecha y condiciones climáticas
def predecir_nivel_lago(año, dia, mes, hora, temperatura, precipitacion, humedad, direccion_viento, velocidad_viento):
    nueva_fecha = pd.DataFrame({
        'AÑO': [año],
        'DIA': [dia],
        'MES': [mes],
        'HORA': [hora],
        'TEMPERATURA (°C)': [temperatura],
        'PRECIPITACION (mm/hora)': [precipitacion],
        'HUMEDAD (%)': [humedad],
        'DIRECCION DEL VIENTO (°)': [direccion_viento],
        'VELOCIDAD DEL VIENTO (m/s)': [velocidad_viento]
    })
    
    nivel_predicho = modelo.predict(nueva_fecha)
    return nivel_predicho[0]


#1/02/2024	02:00	10.7	0	75	86	1.6	3808.14	3808.14

#28/11/2023	10:00	12.8	0	33	110	2.3	3807.97

nivel_predicho = predecir_nivel_lago(2023, 28, 11, 10, 12.8, 0, 33, 110, 2.3)
print(f"Nivel del lago predicho: {nivel_predicho}")

print("--------------------------------")

print(nivel_predicho-error_cuadratico_medio)
print(nivel_predicho+error_cuadratico_medio)