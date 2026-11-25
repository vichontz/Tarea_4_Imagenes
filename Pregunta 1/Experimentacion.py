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


    fig, axes = plt.subplots(2, NIVELES + 1, figsize=(18, 6.5), constrained_layout=True)
    vmax_l = np.percentile(np.abs(np.concatenate([d.ravel() for d in d_lap])), 99.5)
    vmax_s = np.percentile(np.abs(np.concatenate([w.ravel() for w in w_star])), 99.5)

    for j in range(NIVELES):
        # Piramide Laplaciana
        ax_l = axes[0, j]
        ax_l.imshow(d_lap[j], cmap="seismic", vmin=-vmax_l, vmax=vmax_l)
        ax_l.set_title(f"LP d_{j+1}\n{d_lap[j].shape}", fontsize=9)
        ax_l.axis("off")

        # Starlet
        ax_s = axes[1, j]
        ax_s.imshow(w_star[j], cmap="seismic", vmin=-vmax_s, vmax=vmax_s)
        ax_s.set_title(f"Starlet w_{j+1}\n{w_star[j].shape}", fontsize=9)
        ax_s.axis("off")
