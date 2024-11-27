import pandas as pd

# Cargar ambos datasets
dataset1 = pd.read_csv('datos_mensuales_estaciones_1982-2024.csv', delimiter=',', encoding='latin1')  # Ajusta delimitador y codificación
dataset2 = pd.read_csv('Nivel_del_lago.csv', delimiter=',', encoding='latin1')  # Ajusta delimitador y codificación

# Renombrar las columnas para eliminar espacios innecesarios en dataset2
dataset2.columns = dataset2.columns.str.strip()

# Verificar si hay filas con encabezados incorrectos y eliminarlas
dataset2 = dataset2[~dataset2['FECHA (AÑO/MES/DIA)'].str.contains('AÑO', na=False)]

# Renombrar las columnas de fechas para que coincidan
dataset1 = dataset1.rename(columns={'fecha': 'FECHA'})
dataset2 = dataset2.rename(columns={'FECHA (AÑO/MES/DIA)': 'FECHA'})

# Convertir las fechas al mismo formato
dataset1['FECHA'] = pd.to_datetime(dataset1['FECHA'], format='%Y-%m-%d')
dataset2['FECHA'] = pd.to_datetime(dataset2['FECHA'], format='%Y/%m/%d', errors='coerce')  # 'coerce' pone NaT en caso de error

# Eliminar filas con fechas no válidas
dataset2 = dataset2.dropna(subset=['FECHA'])

# Realizar la combinación basada en la columna 'FECHA'
dataset_combinado = pd.merge(dataset2, dataset1[['FECHA', 'dato']], on='FECHA', how='left')

# Guardar el nuevo dataset combinado
dataset_combinado.to_csv('dataset_combinado.csv', index=False, sep=',')
