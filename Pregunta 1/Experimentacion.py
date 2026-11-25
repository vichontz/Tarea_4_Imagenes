#Experimentación y análisis (10 puntos) 



#2. Aplique umbral duro a las capas de detalle. Para cada capa use Tⱼ = α·max|dⱼ| (umbral relativo) y pruebe al 
#menos dos valores de α comunes a ambas representaciones. Reporte el porcentaje de coeficientes 
#conservados por capa, reconstruya la imagen y analice pérdida de detalle, aparición de artefactos y 
#diferencias entre ambos métodos.
 
#3. Modifique las capas mediante factores multiplicativos generales. Pruebe al menos dos configuraciones: 
#una que enfatice principalmente estructuras finas y otra que enfatice estructuras de mayor escala. Use el 
#mismo criterio cualitativo de ganancias en ambas representaciones y reconstruya las imágenes resultantes. 
#Compare visualmente ambas alternativas y muestre al menos un recorte ampliado donde las diferencias sean 
#claras. 



import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Implementacion import cargar_luminancia, compresion_global, piramide_laplaciana, reconstruir_piramide, starlet, reconstruir_starlet, rmse

raiz = Path(__file__).resolve().parent.parent
RUTA = Path(sys.argv[1]) if len(sys.argv) > 1 else raiz / "archivos_t4" / "memorial.tif"
carpeta = Path(__file__).resolve().parent / "Resultados"
carpeta.mkdir(exist_ok=True)
NIVELES = 5

#Definiciones globales
Y = np.maximum(cargar_luminancia(RUTA), 0.0)
I = compresion_global(Y) 
d_lap, c_lap = piramide_laplaciana(I, NIVELES)
w_star, c_star = starlet(I, NIVELES)


#1. Muestre las cinco capas de detalle y el residual grueso de ambas representaciones. Para facilitar la 
#comparación, utilice escalas de visualización coherentes e indique las dimensiones de cada capa. 

def experimento_1():
    return
