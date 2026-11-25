import numpy as np
from scipy.ndimage import convolve1d

import tifffile
import sys

from pathlib import Path
import matplotlib.pyplot as plt

B3 = np.array([1, 4, 6, 4, 1]) / 16
 
def suavizar(c, h):
    c = convolve1d(c, h, axis=0, mode="mirror")
    return convolve1d(c, h, axis=1, mode="mirror")


def reducir(c):
    return suavizar(c, B3)[::2, ::2]


def expandir(c, forma):
    z = np.zeros(forma)
    z[::2, ::2] = c
    return 4 * suavizar(z, B3)


def piramide_laplaciana(imagen, niveles=5):
    #([d_1..d_niveles], c_niveles) con d_j = c_{j-1} - U(c_j).
    detalles, c = [], imagen
    for _ in range(niveles):
        c_sig = reducir(c)
        detalles.append(c - expandir(c_sig, c.shape))
        c = c_sig
    return detalles, c


def reconstruir_piramide(detalles, residuo):
    #c_{j-1} = d_j + U(c_j), desde el nivel mas grueso hasta el original.
    c = residuo
    for d in reversed(detalles):
        c = d + expandir(c, d.shape)
    return c


def starlet(imagen, escalas=5):
   #Devuelve ([w_1..w_escalas], c_escalas), w_j = c_{j-1} - c_j.
    capas, c = [], imagen
    for j in range(escalas):
        h = np.zeros(4 * 2**j + 1)
        h[:: 2**j] = B3
        c_sig = suavizar(c, h)
        capas.append(c - c_sig)
        c = c_sig
    return capas, c


def reconstruir_starlet(capas, residuo):
    return residuo + sum(capas)



def cargar_luminancia(ruta, canales=3):
    rgb = tifffile.imread(ruta).astype(np.float64)[..., :canales]
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def compresion_global(Y, clave=0.18, eps=1e-6):
    Ys = clave * Y / np.exp(np.mean(np.log(Y + eps)))
    L = Ys / (1 + Ys)
    return (L - L.min()) / (L.max() - L.min())


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))



    
if __name__ == "__main__":
    
    raiz = Path(__file__).resolve().parent.parent
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else raiz / "archivos_t4" / "memorial.tif"
    carpeta = Path(__file__).resolve().parent / "Resultados"
    carpeta.mkdir(exist_ok=True)

    Y = cargar_luminancia(ruta)
    I = compresion_global(Y, clave=0.18)

    # Descomposicion y reconstruccion
    d_lap, c_lap = piramide_laplaciana(I, 5)
    w_star, c_star = starlet(I, 5)
    print(f"RMSE piramide Laplaciana: {rmse(I, reconstruir_piramide(d_lap, c_lap)):.3e}")
    print(f"RMSE starlet:             {rmse(I, reconstruir_starlet(w_star, c_star)):.3e}")

    
    fig, ax = plt.subplots(figsize=(5, 7.5))
    ax.imshow(I, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{ruta.name}: luminancia, Reinhard global (clave=0.18)\n{I.shape[0]}x{I.shape[1]}")
    ax.axis("off")
    fig.savefig(carpeta/"luminancia_comprimida.png", dpi=250, bbox_inches="tight")
    print("Figura guardada en", carpeta / "luminancia_comprimida.png")
    plt.show()