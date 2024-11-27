# Casos de problemas cardiacos de acuerdo a la edad

''' 
    Documentacion de Adair Vidal para el analisis de datos en base a la interpolacion de estos mismos
        1. DataSets obtenidos de: https://archive.ics.uci.edu/dataset/45/heart+disease
        2. Los .data que no estan procesados son dificiles de manejar y el unico bien documentado es el de cleveland
        3. La columna class se refiere a la clasificacion de la enfermedad cardiaca, esta ordenada numericamente
        4. Por el tipo de planteamiento del problema es mejor manejar las cantidades de edad y enfermedad en enteros 
            redondeados ya que es inutil o rebuscado medir las fracciones de las edades y cantidades de casos.
        5. Como no hay una medida estandar para interpolar valores, estoy interpolando la misma cantidad de datos en x
            que existen de forma unica (edad maxima menos la edad minima)
        6. Para determinar cual es el mas adecuado estoy usando las medidas de Error Cuadratico medio (MSE en ingles),
            y el Coeficiente de determinacion (Simbolizado por R^2). Los cuales indican que entre menos MSE y un R^2 mas
            cercano a 1 son mejores objetivamente.
        7. Figura: X = Edad, Y = Casos de Enfermedad, 
            Linea Roja = Interpolacion Lineal
            Linea Azul = Interpolacion Cuadratica
            Linea Verde = Interpolacion Cubica
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score
import os

def edad_Enfermedad():
    # File del DataSet
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cleveland_data = 'DataSets/heart+disease/processed.cleveland.data'
    # Nombre de Columnas en base al mod de John Gennari
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
               "oldpeak", "slope", "ca", "thal", "class"]

    # Creacion del dataframe con los nombres de columnas y validacion de campos nulos
    df = pd.read_csv(cleveland_data, header=None, names=columns, na_values=["?"])

    # Obtener un dataframe que contenga unicamente la edad y el diagnostico de la enfermedad
    edad_Enfermedad = df[['age', 'class']]

    # Limpiar el dataframe de los valores nulos
    edad_Enfermedad = edad_Enfermedad.dropna(subset=['age', 'class'])

    # Añade una columna que determina si un paciente tiene una enfermedad cardiaca y les asigna valor binario
    edad_Enfermedad['heart_disease'] = edad_Enfermedad['class'].apply(lambda x: 1 if x > 0 else 0)  # Lambda al parecer permite crear funciones anonimas sencillas

    # Agrupa y recuenta en una variable por edad la cantidad de veces que han tenido enfermedad cardiaca
    age_grouped = edad_Enfermedad.groupby('age')['heart_disease'].sum()

    # Variables  para interpolacion y graficacion
    x = age_grouped.index.values  # Tag de edad
    y = age_grouped.values  # Recuento de enfermedad cardiaca por edad

    # Variable de X para interpolar n valores equidistantes entre el minimo y el maximo encontrado en el dataset
    n = (max(x)-min(x)).astype(int)
    x_new = np.linspace(min(x), max(x), n)
    # Elimina valores repetidos
    x_new = np.unique(x_new)

    # Interpolacion lineal
    linear_interp = interp1d(x, y, kind='linear') # Objeto de interpolacion de los valores originales
    y_linear = linear_interp(x_new) # Variable de almacenamiento de todos los valores a interpolar

    # Interpolacion cuadratica
    quadratic_interp = interp1d(x, y, kind='quadratic')
    y_quadratic = quadratic_interp(x_new)

    # Interpolacion cubica
    cubic_interp = interp1d(x, y, kind='cubic')
    y_cubic = cubic_interp(x_new)

    # Variable de Y Original del mismo tamaño que la interpolación
    y_real = [age_grouped.get(age.astype(int), 0) for age in x_new] # Rellena de 0 los valores de x que no tenga

    # Calculo de MSE para cada interpolacion
    mse_linear = mean_squared_error(y_real, y_linear)
    mse_quadratic = mean_squared_error(y_real, y_quadratic)
    mse_cubic = mean_squared_error(y_real, y_cubic)

    # Calculo de R^2 para cada interpolacion
    r2_linear = r2_score(y_real, y_linear)
    r2_quadratic = r2_score(y_real, y_quadratic)
    r2_cubic = r2_score(y_real, y_cubic)

    # Error Cuadratico
    print(f"MSE Lineal:", mse_linear)
    print(f"MSE Cuadratica:", mse_quadratic)
    print(f"MSE Cubica:", mse_cubic)
    # Coeficiente de determinacion
    print("\nR^2 Lineal:", r2_linear)
    print("R^2 Cuadratica:", r2_quadratic)
    print("R^2 Cubica:", r2_cubic)

    # DataFrame para guardar resultados
    resultados_interp = pd.DataFrame({
        'Edad': x_new.round().astype(int),
        'Diagnosticos Originales': y_real,
        'Diagnostico Interpolado (Lineal)': y_linear.astype(int),
        'Diagnostico Interpolado (Cuadratica)': y_quadratic.astype(int),
        'Diagnostico Interpolado (Cubica)': y_cubic.astype(int)
    })

    resultados_originales = pd.DataFrame({
        'Edad': x,
        'Diagnosticos Originales': y,
    })
    resultados_originales.to_csv('Resultados/Enfermedad-Edad.csv', index=False)

    # Objeto de graficacion
    plt.figure(figsize=(12, 8))

    # Graficacion de las interpolaciones
    plt.plot(x_new, y_linear, label='Interpolacion Lineal', color='red', linestyle='dotted')
    plt.plot(x_new, y_quadratic, label='Interpolacion Cuadratica', color='blue', linestyle='dotted')
    plt.plot(x_new, y_cubic, label='Interpolacion Cubica', color='green', linestyle='dotted')

    # Mostrar los puntos del dataset originales
    plt.scatter(x, y, color='gray', label='Datos Originales')
    # Mostrar las etiquetas de Edad de 5 en 5, empezando por el 30
    plt.xticks(np.arange(min(x)+1, max(x), 5))
    # Establece que el eje Y solo mostrara los valores arriba de 0 en la grafica
    # plt.gca().set_ylim(bottom=0)
    # Ajuste del eje de Y para que muestre una grafica mas bonita
    plt.yticks(np.arange(max(y)*-1+2, max(y)+2, 2))

    # Nombres y etiquetas
    plt.title('Interpolacion de la Cantidad de Diagnosticos de Enfermedad Cardíaca por Edad')
    plt.xlabel('Edad')
    plt.ylabel('Cantidad de Diagnosticos')
    plt.legend()

    # Mostrar la grafica
    plt.grid(True)
    plt.show()

    # Mostrar y guardar los resultados en CSV
    print('\n',resultados_interp)
    resultados_interp.to_csv('Resultados/Enfermedad-Edad_Interpolacion.csv', index=False)
    return ()