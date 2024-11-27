import pandas as pd
from pandas.tseries.offsets import MonthEnd

# Cargar el dataset combinado
dataset_combinado = pd.read_csv('dataset_combinado.csv', delimiter=',', encoding='latin1')

# Asegurarnos de que la columna 'FECHA' esté en formato de fecha
dataset_combinado['FECHA'] = pd.to_datetime(dataset_combinado['FECHA'])

# Crear una nueva columna para el nivel del lago
dataset_combinado['nivel_lago'] = None

# Filtrar el dataset para obtener el nivel del lago solo para el día 1 de cada mes
niveles_dia_1 = dataset_combinado[dataset_combinado['FECHA'].dt.day == 1][['FECHA', 'dato']]

# Iterar sobre cada fila en niveles_dia_1 y aplicar el nivel de lago a todo el mes
for index, row in niveles_dia_1.iterrows():
    fecha_inicio = row['FECHA']
    nivel_lago = row['dato']
    
    # Crear un rango de fechas para el mes
    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_inicio + MonthEnd(0))
    
    # Actualizar la columna 'nivel_lago' en el dataset combinado
    dataset_combinado.loc[dataset_combinado['FECHA'].isin(rango_fechas), 'nivel_lago'] = nivel_lago

# Guardar el dataset actualizado con los niveles de lago completados
dataset_combinado.to_csv('dataset_combinado_actualizado.csv', index=False, sep=',')
