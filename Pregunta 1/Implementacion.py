import numpy as np
from scipy.ndimage import convolve1d

import tifffile
import sys
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
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    RUTA = sys.argv[1] if len(sys.argv) > 1 else "archivos_t4/memorial.tif"
    NIVELES = 5

    
    Y = cargar_luminancia(RUTA)
    I = compresion_global(Y)
    print(f"Imagen: {RUTA}  forma={I.shape}")
    print(f"Luminancia HDR: min={Y.min():.4g} max={Y.max():.4g}  ->  comprimida: [{I.min():.2f}, {I.max():.2f}]")

    
    d_lap, c_lap = piramide_laplaciana(I, NIVELES)
    w_star, c_star = starlet(I, NIVELES)

    #Reconstruimos
    I_lap = reconstruir_piramide(d_lap, c_lap)
    I_star = reconstruir_starlet(w_star, c_star)
    print(f"\nRMSE reconstruccion piramide Laplaciana: {rmse(I, I_lap):.3e}")
    print(f"RMSE reconstruccion starlet:             {rmse(I, I_star):.3e}")

    
    print("\nPiramide Laplaciana:")
    for j, d in enumerate(d_lap, 1):
        print(f"  d_{j}: {d.shape}")
    print(f"  residuo c_{NIVELES}: {c_lap.shape}")
    n_lap = sum(d.size for d in d_lap) + c_lap.size
    print(f"  coeficientes / pixeles = {n_lap / I.size:.3f}")
    print(f"Starlet: {NIVELES} capas + residuo, todas {I.shape}  ->  coeficientes / pixeles = {NIVELES + 1}")
    