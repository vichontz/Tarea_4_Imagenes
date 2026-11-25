#Experimentación y análisis (10 puntos) 

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyparsing import alphas
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

        # Residuos
    axes[0, NIVELES].imshow(c_lap, cmap="gray", vmin=0, vmax=1)
    axes[0, NIVELES].set_title(f"LP Residuo c_{NIVELES}\n{c_lap.shape}", fontsize=9)
    axes[0, NIVELES].axis("off")

    axes[1, NIVELES].imshow(c_star, cmap="gray", vmin=0, vmax=1)
    axes[1, NIVELES].set_title(f"Starlet Residuo c_{NIVELES}\n{c_star.shape}", fontsize=9)
    axes[1, NIVELES].axis("off")

    fig.suptitle(
        f"Capas de detalle (LP ±{vmax_l:.3f}, Starlet ±{vmax_s:.3f}) y residual",
        fontsize=12,
    )
    plt.savefig(carpeta/"p1_exp1_capas.png", dpi=200, bbox_inches="tight")
    plt.show()

#2. Aplique umbral duro a las capas de detalle. Para cada capa use Tⱼ = α·max|dⱼ| (umbral relativo) y pruebe al 
#menos dos valores de α comunes a ambas representaciones. Reporte el porcentaje de coeficientes 
#conservados por capa, reconstruya la imagen y analice pérdida de detalle, aparición de artefactos y 
#diferencias entre ambos métodos.


def umbral_duro(capas, alpha):
    capas_thr = []
    porcentajes = []
    for c in capas:
        T_j = alpha * np.max(np.abs(c))
        c_mod = np.where(np.abs(c) >= T_j, c, 0.0)
        pct = 100.0 * np.count_nonzero(c_mod) / c.size
        capas_thr.append(c_mod)
        porcentajes.append(pct)
    return capas_thr, porcentajes


def experimento_2():

    alphas = (0.05, 0.20)  # Aca podemos ir cambiando los alphas
    recorte = (slice(430, 560), slice(295, 405))

    fig_g, axes_g = plt.subplots(len(alphas), 2, figsize=(10, 7.5), constrained_layout=True)
    fig_r, axes_r = plt.subplots(len(alphas), 2, figsize=(10, 7.5), constrained_layout=True)

    for f, alpha in enumerate(alphas):
        d_t, pct_l = umbral_duro(d_lap, alpha)
        w_t, pct_s = umbral_duro(w_star, alpha)

        rec_l = np.clip(reconstruir_piramide(d_t, c_lap), 0, 1)
        rec_s = np.clip(reconstruir_starlet(w_t, c_star), 0, 1)

        rmse_l = rmse(I, rec_l)
        rmse_s = rmse(I, rec_s)

        print(f"\nalpha = {alpha:.2f}:")
        print(f"  LP       % conservado : {[round(p, 2) for p in pct_l]},  RMSE = {rmse_l:.4f}")
        print(f"  Starlet  % conservado : {[round(p, 2) for p in pct_s]},   RMSE = {rmse_s:.4f}")

        axes_g[f, 0].imshow(rec_l, cmap="gray", vmin=0, vmax=1)
        axes_g[f, 0].set_title(f"LP (alpha={alpha})\nRMSE={rmse_l:.4f}", fontsize=9)
        axes_g[f, 0].axis("off")

        axes_g[f, 1].imshow(rec_s, cmap="gray", vmin=0, vmax=1)
        axes_g[f, 1].set_title(f"Starlet (alpha={alpha})\nRMSE={rmse_s:.4f}", fontsize=9)
        axes_g[f, 1].axis("off")

        axes_r[f, 0].imshow(rec_l[recorte], cmap="gray", vmin=0, vmax=1)
        axes_r[f, 0].set_title(f"Recorte LP (alpha={alpha})", fontsize=9)
        axes_r[f, 0].axis("off")

        axes_r[f, 1].imshow(rec_s[recorte], cmap="gray", vmin=0, vmax=1)
        axes_r[f, 1].set_title(f"Recorte Starlet (alpha={alpha})", fontsize=9)
        axes_r[f, 1].axis("off")

    fig_g.suptitle("Umbral duro y artefactos - imágenes globales", fontsize=12)
    fig_r.suptitle("Umbral duro y artefactos - recortes", fontsize=12)
    fig_g.savefig(carpeta / "p1_exp2_umbral_globales.png", dpi=200, bbox_inches="tight")
    fig_r.savefig(carpeta / "p1_exp2_umbral_recortes.png", dpi=200, bbox_inches="tight")
    plt.show()




 
#3. Modifique las capas mediante factores multiplicativos generales. Pruebe al menos dos configuraciones: 
#una que enfatice principalmente estructuras finas y otra que enfatice estructuras de mayor escala. Use el 
#mismo criterio cualitativo de ganancias en ambas representaciones y reconstruya las imágenes resultantes. 
#Compare visualmente ambas alternativas y muestre al menos un recorte ampliado donde las diferencias sean 
#claras. 
def experimento_3():

    # Para realzar detalles, hay que aumentar la ganancia de las capas de detalle finas (j=1,2)
    # y disminuir la ganancia de las capas gruesas (j=4,5). Y Viceversa

    g_fino = [2.5, 1.8, 1.2, 1.0, 1.0]
    g_grueso = [1.0, 1.0, 1.2, 1.8, 2.5]

    recorte1 = (slice(430, 560), slice(295, 405))
    recorte2 = (slice(525, 645), slice(65, 220))
    configs = [
        ("Enfasis Fino " + str(g_fino), g_fino),
        ("Enfasis Grueso " + str(g_grueso), g_grueso),
    ]

    fig_g, axes_g = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    

    for row, (nombre, gains) in enumerate(configs):
        d_mod = [g * d for g, d in zip(gains, d_lap)]
        w_mod = [g * w for g, w in zip(gains, w_star)]

        rec_lap = np.clip(reconstruir_piramide(d_mod, c_lap), 0, 1)
        rec_star = np.clip(reconstruir_starlet(w_mod, c_star), 0, 1)

        axes_g[row, 0].imshow(rec_lap, cmap="gray", vmin=0, vmax=1)
        axes_g[row, 0].set_title(f"LP - {nombre}", fontsize=8.5)
        axes_g[row, 0].axis("off")

        axes_g[row, 1].imshow(rec_star, cmap="gray", vmin=0, vmax=1)
        axes_g[row, 1].set_title(f"Starlet - {nombre}", fontsize=8.5)
        axes_g[row, 1].axis("off")

        

    fig_g.suptitle("Ganancia por escala - imágenes globales", fontsize=12)
    fig_g.savefig(carpeta / "p1_exp3_ganancias_globales.png", dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
# Comentar y descomentar segun lo que se quiera ejecutar
    #experimento_1()
    #experimento_2()
    experimento_3()