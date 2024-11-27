import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos con manejo de errores
def cargar_datos(ruta):
    try:
        df = pd.read_csv(ruta, delimiter=';', encoding='latin1')
        print("Datos cargados exitosamente.")
        return df
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {ruta}")
        return None

# Preprocesamiento de datos
def preprocesar_datos(df):
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Convertir columna de fecha
    df['FECHA_HORA'] = pd.to_datetime(df['FECHA'] + ' ' + df['HORA'], format='%d/%m/%Y %H:%M')
    
    # Extraer características temporales
    df['AÑO'] = df['FECHA_HORA'].dt.year
    df['DIA'] = df['FECHA_HORA'].dt.day
    df['MES'] = df['FECHA_HORA'].dt.month
    df['HORA'] = df['FECHA_HORA'].dt.hour
    
    # Convertir columnas a numéricas
    columnas_numericas = [
        'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 'HUMEDAD (%)', 
        'DIRECCION DEL VIENTO (°)', 'VELOCIDAD DEL VIENTO (m/s)', 'NIVEL_LAGO'
    ]
    
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con valores nulos
    df.dropna(subset=columnas_numericas, inplace=True)
    
    # Eliminar columnas innecesarias
    df.drop(['FECHA_HORA', 'FECHA', 'HORA'], axis=1, inplace=True)
    
    return df

# Análisis exploratorio de datos
def analisis_exploratorio(df):
    # Matriz de correlación
    plt.figure(figsize=(10, 8))
    correlacion = df[['AÑO', 'DIA', 'MES', 
                      'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 
                      'HUMEDAD (%)', 'DIRECCION DEL VIENTO (°)', 
                      'VELOCIDAD DEL VIENTO (m/s)', 'NIVEL_LAGO']].corr()
    
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlación de Variables')
    plt.tight_layout()
    plt.show()

# Entrenar modelos
def entrenar_modelos(X_train, X_test, y_train, y_test, X, y):
    # Create pipeline with scaling and different models
    modelos = {
        'Regresión Lineal': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    
    resultados = {}
    mejor_modelo = None
    mejor_r2 = -float('inf')
    
    for nombre, modelo in modelos.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', modelo)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        
        resultados[nombre] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'CV_MSE': -cv_scores.mean()
        }
        
        # Track the best model based on R2 score
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = pipeline
    
    return resultados, mejor_modelo

# Función de predicción
def predecir_nivel_lago(modelo, año, dia, mes, temperatura, precipitacion, humedad, direccion_viento, velocidad_viento):
    nueva_fecha = pd.DataFrame({
        'AÑO': [año],
        'DIA': [dia],
        'MES': [mes],
        'TEMPERATURA (°C)': [temperatura],
        'PRECIPITACION (mm/hora)': [precipitacion],
        'HUMEDAD (%)': [humedad],
        'DIRECCION DEL VIENTO (°)': [direccion_viento],
        'VELOCIDAD DEL VIENTO (m/s)': [velocidad_viento]
    })
    
    nivel_predicho = modelo.predict(nueva_fecha)
    return nivel_predicho[0]

# Flujo principal
def main():
    # Cargar datos
    df = cargar_datos('DATA LAGO.csv')
    
    if df is not None:
        # Preprocesar datos
        df = preprocesar_datos(df)
        
        # Análisis exploratorio
        analisis_exploratorio(df)
        
        # Preparar datos para machine learning
        # Prepare data for machine learning
        X = df[['AÑO', 'DIA', 'MES', 
                'TEMPERATURA (°C)', 'PRECIPITACION (mm/hora)', 
                'HUMEDAD (%)', 'DIRECCION DEL VIENTO (°)', 
                'VELOCIDAD DEL VIENTO (m/s)']]
        y = df['NIVEL_LAGO']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models (now passing X and y for cross-validation)
        resultados, mejor_modelo = entrenar_modelos(X_train, X_test, y_train, y_test, X, y)
        
        # Imprimir resultados
        for nombre, metricas in resultados.items():
            print(f"\nResultados {nombre}:")
            for metrica, valor in metricas.items():
                print(f"{metrica}: {valor}")
        
        # Ejemplo de predicción
        prediccion = predecir_nivel_lago(
            mejor_modelo, 
            año=2023, 
            dia=28, 
            mes=11, 
            temperatura=12.8, 
            precipitacion=0, 
            humedad=33, 
            direccion_viento=110, 
            velocidad_viento=2.3
        )
        
        print(f"\nNivel del Lago Predicho: {prediccion} metros")

if __name__ == "__main__":
    main()