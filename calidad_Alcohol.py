'''
    Documentacion de Adair Vidal para experimento de interpolacion usando los datasets de dos tipos de vino,
    en base al % de alcohol de ambos y su calidad.

    Dataset obtenido de: https://archive.ics.uci.edu/dataset/186/wine+quality
        P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
        Modeling wine preferences by data mining from alcoholysicochemical properties.
        In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

    1.  Se toman la interpolacion del vino rojo y del vino blanco como dos ejercicios separados para ver su
        comparacion.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score
import os

def calidad_alcohol_rojo():
    # Asegurar de que la ruta de acceso a los archivos sea directa independientemente del OS
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = 'DataSets/wine+quality/winequality-red.csv'
    # DataFrame de acuerdo al csv de datos del vino rojo
    df = pd.read_csv(data, header=0, delimiter=';') # Aqui el delimitador del csv es ';' en lugar de ','

    # Seleccion de categorias a utilizar
    alcohol_calidad = df[['alcohol','quality']]

    # Media de vinos en funcion de su calidad dependiendo del alcohol
    calidad_grouped = alcohol_calidad.groupby('alcohol')['quality'].mean()
    # print(calidad_grouped) # DEBUG

    # Variables para graficacion e interpolacion
    x = calidad_grouped.index.values  # alcohol como eje X
    y = calidad_grouped.values  # Media de calidad como eje Y

    # Intervalo equidistante a interpolar
    x_new =  np.linspace(min(x), max(x), 100)
    # print(x_new) # DEBUG

    # Interpolación lineal
    linear_interp = interp1d(x, y, kind='linear')
    y_linear = linear_interp(x_new)
    y_linear = np.clip(y_linear, 0, 10)

    # Interpolación cuadrática
    quadratic_interp = interp1d(x, y, kind='quadratic')
    y_quadratic = quadratic_interp(x_new)
    y_quadratic = np.clip(y_quadratic, 0, 10)

    # Interpolación cúbica
    cubic_interp = interp1d(x, y, kind='cubic')
    y_cubic = cubic_interp(x_new)
    y_cubic = np.clip(y_cubic, 0, 10)

    # Calculo de MSE para cada interpolacion
    mse_linear = mean_squared_error(y, y_linear[:len(y)])
    mse_quadratic = mean_squared_error(y, y_quadratic[:len(y)])
    mse_cubic = mean_squared_error(y, y_cubic[:len(y)])
    # Calculo de R^2 para cada interpolacion
    r2_linear = r2_score(y, y_linear[:len(y)])
    r2_quadratic = r2_score(y, y_quadratic[:len(y)])
    r2_cubic = r2_score(y, y_cubic[:len(y)])
    # Calcular Error Estándar de Estimación (SEE)
    see_linear = error_estandar_estimacion(y, y_linear[:len(y)])
    see_quadratic = error_estandar_estimacion(y, y_quadratic[:len(y)])
    see_cubic = error_estandar_estimacion(y, y_cubic[:len(y)])

    # Error Estándar de Estimación (SEE)
    print("\nSEE Lineal:", see_linear)
    print("SEE Cuadrática:", see_quadratic)
    print("SEE Cúbica:", see_cubic)
    # Error Cuadratico
    print(f"\nMSE Lineal:", mse_linear)
    print(f"MSE Cuadratica:", mse_quadratic)
    print(f"MSE Cubica:", mse_cubic)
    # Coeficiente de determinacion
    print("\nR^2 Lineal:", r2_linear)
    print("R^2 Cuadratica:", r2_quadratic)
    print("R^2 Cubica:", r2_cubic)

    resultados_interp = pd.DataFrame({
        'Alcohol': np.round(x_new, 2),
        'Calidad (Lineal)': y_linear,
        'Calidad (Cuadratica)': y_quadratic,
        'Calidad (Cubica)': y_cubic
    })

    resultados_originales = pd.DataFrame({
        'Alcohol': x,
        'Calidad': y,
    })
    resultados_originales.to_csv('Resultados/Alcohol-Calidad-Rojo.csv', index=False)

    # Graficacion
    plt.figure(figsize=(12, 8))

    # Graficacion de las interpolaciones
    plt.plot(x_new, y_linear, label='Interpolacion Lineal', color='red', linestyle='dotted')
    plt.plot(x_new, y_quadratic, label='Interpolacion Cuadratica', color='blue', linestyle='dotted')
    plt.plot(x_new, y_cubic, label='Interpolacion Cubica', color='green', linestyle='dotted')

    # Mostrar los puntos del dataset originales
    plt.scatter(x, y, color='gray', label='Datos Originales')

    # Ejes modificados
    plt.xticks(np.arange(min(x)+0.1, max(x)+0.4, 0.5))
    plt.yticks(np.arange(min(y)-1, 11, 1))

    # Nombres y etiquetas
    plt.title('Calidad del vino rojo respecto al alcohol')
    plt.xlabel('alcohol')
    plt.ylabel('Calidad')
    plt.legend()

    # Mostrar la grafica
    plt.grid(True)
    plt.show()

    #Guardar el CSV
    print('\n', resultados_interp)
    resultados_interp.to_csv('Resultados/alcohol-Calidad-Rojo_Interpolacion.csv', index=False)
    return()

def calidad_alcohol_blanco():
    # Asegurar de que la ruta de acceso a los archivos sea directa independientemente del OS
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = 'DataSets/wine+quality/winequality-white.csv'
    # DataFrame de acuerdo al csv de datos del vino rojo
    df = pd.read_csv(data, header=0, delimiter=';') # Aqui el delimitador del csv es ';' en lugar de ','

    # Seleccion de categorias a utilizar
    alcohol_calidad = df[['alcohol','quality']]

    # Media de vinos en funcion de su calidad dependiendo del alcohol
    calidad_grouped = alcohol_calidad.groupby('alcohol')['quality'].mean()
    # print(calidad_grouped) # DEBUG

    # Variables para graficacion e interpolacion
    x = calidad_grouped.index.values  # alcohol como eje X
    y = calidad_grouped.values  # Media de calidad como eje Y

    # Intervalo equidistante a interpolar
    x_new =  np.linspace(min(x), max(x), 110)
    # print(x_new) # DEBUG

    # Interpolación lineal
    linear_interp = interp1d(x, y, kind='linear')
    y_linear = linear_interp(x_new)
    y_linear = np.clip(y_linear, 0, 10)

    # Interpolación cuadrática
    quadratic_interp = interp1d(x, y, kind='quadratic')
    y_quadratic = quadratic_interp(x_new)
    y_quadratic = np.clip(y_quadratic, 0, 10)

    # Interpolación cúbica
    cubic_interp = interp1d(x, y, kind='cubic')
    y_cubic = cubic_interp(x_new)
    y_cubic = np.clip(y_cubic, 0, 10)

    # Calculo de MSE para cada interpolacion
    mse_linear = mean_squared_error(y, y_linear[:len(y)])
    mse_quadratic = mean_squared_error(y, y_quadratic[:len(y)])
    mse_cubic = mean_squared_error(y, y_cubic[:len(y)])
    # Calculo de R^2 para cada interpolacion
    r2_linear = r2_score(y, y_linear[:len(y)])
    r2_quadratic = r2_score(y, y_quadratic[:len(y)])
    r2_cubic = r2_score(y, y_cubic[:len(y)])
    # Calcular Error Estándar de Estimación (SEE)
    see_linear = error_estandar_estimacion(y, y_linear[:len(y)])
    see_quadratic = error_estandar_estimacion(y, y_quadratic[:len(y)])
    see_cubic = error_estandar_estimacion(y, y_cubic[:len(y)])

    # Error Estándar de Estimación (SEE)
    print("\nSEE Lineal:", see_linear)
    print("SEE Cuadrática:", see_quadratic)
    print("SEE Cúbica:", see_cubic)
    # Error Cuadratico
    print(f"\nMSE Lineal:", mse_linear)
    print(f"MSE Cuadratica:", mse_quadratic)
    print(f"MSE Cubica:", mse_cubic)
    # Coeficiente de determinacion
    print("\nR^2 Lineal:", r2_linear)
    print("R^2 Cuadratica:", r2_quadratic)
    print("R^2 Cubica:", r2_cubic)

    # DataFrame para guardar resultados de las interpolaciones
    resultados_interp = pd.DataFrame({
        'Alcohol': np.round(x_new, 2),
        'Calidad (Lineal)': y_linear,
        'Calidad (Cuadratica)': y_quadratic,
        'Calidad (Cubica)': y_cubic
    })

    resultados_originales = pd.DataFrame({
        'Alcohol': x,
        'Calidad': y,
    })
    resultados_originales.to_csv('Resultados/Alcohol-Calidad-Blanco.csv', index=False)

    # Graficacion
    plt.figure(figsize=(12, 8))

    # Graficacion de las interpolaciones
    plt.plot(x_new, y_linear, label='Interpolacion Lineal', color='red', linestyle='dotted')
    plt.plot(x_new, y_quadratic, label='Interpolacion Cuadratica', color='blue', linestyle='dotted')
    plt.plot(x_new, y_cubic, label='Interpolacion Cubica', color='green', linestyle='dotted')

    # Mostrar los puntos del dataset originales
    plt.scatter(x, y, color='gray', label='Datos Originales')

    # Ejes modificados
    plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
    plt.yticks(np.arange(min(y)-3, 11, 1))

    # Nombres y etiquetas
    plt.title('Calidad del vino blanco respecto al alcohol')
    plt.xlabel('alcohol')
    plt.ylabel('Calidad')
    plt.legend()

    # Mostrar la grafica
    plt.grid(True)
    plt.show()

    #Guardar el CSV
    print('\n', resultados_interp)
    resultados_interp.to_csv('Resultados/alcohol-Calidad-Blanco_Interpolacion.csv', index=False)
    return()

def error_estandar_estimacion(y_real, y_predicho):
    # Calcular el error estándar de estimación (SEE)
    n = len(y_real)
    residuals = y_real - y_predicho
    see = np.sqrt(np.sum(residuals**2) / (n - 2))  # Dividido por n-2 por el número de parámetros
    return see
