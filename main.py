# main.py
'''
    Adair Vidal:
        Los programas de interpolacion se parecen y no estan bien segmentado en funciones, debido a que el codigo
        para interpolar es casi igual, esto hace que tengan mucha redundancia pero no tengo tiempo de mejorarlo
        asi que preferi hacer ctrl+c, ctrl+v y ctrl+f para reemplazar nombres.

        # Nota mental: cambiar el codigo a funciones dependientes y reutilizar codigo
'''
from edad_enfermedad import edad_Enfermedad
from colesterol_enfermedad import colesterol_Enfermedad
from presion_enfermedad import presion_Enfermedad
from calidad_Ph import calidad_pH_rojo, calidad_pH_blanco
from calidad_Alcohol import calidad_alcohol_rojo, calidad_alcohol_blanco

def main():
    # Descomentar la funcion a utilizar:
    # calidad_pH_rojo()
    # calidad_pH_blanco()
    # calidad_alcohol_blanco()
    # calidad_alcohol_rojo()
    edad_Enfermedad()
    colesterol_Enfermedad()
    presion_Enfermedad()
    return ()

if __name__ == "__main__":
    main()