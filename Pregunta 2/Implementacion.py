
#Implementación (8 puntos) 

#•  Para cada imagen, obtenga una luminancia Y o L, y aplique sobre ella el filtro homomórfico y el método de 
#   Durand usando parámetros base razonables, preferentemente los utilizados en las cápsulas. Mantenga esos 
#   parámetros fijos y aplíquelos a las componentes RGB para usar como imagen de comparación. 

#•  Implemente una primera estrategia de reconstrucción de color a partir de la luminancia procesada Y'. Para 
#   cada canal C ∈ {R,G,B}, utilice una relación de cromaticidad con un parámetro β que controle la fuerza del 
#   color: 
#                                       C' = Y' · ( C / (Y + ε) )ᵝ 

#•  Implemente una segunda estrategia que ajuste explícitamente el color después de la compresión de 
#   luminancia. Puede trabajar sobre la saturación S en HSV/HSI o sobre el croma C* en CIELCh. La 
#   transformación utilizada, sus parámetros y el tratamiento de valores fuera de rango deben quedar explícitos 
#   en el código. 

#•  Evite divisiones por cero y reporte cualquier normalización, clipping o conversión de rango utilizada 
#   durante la reconstrucción. 


import numpy as np
import tifffile




# Funciones varias:

def cargar_rgb_hdr(ruta):
    rgb = tifffile.imread(ruta).astype(np.float64)
    return np.maximum(rgb[..., :3], 0)

def normalizar_rgb(E):
    # Lleva RGB a [0,1] preservando las proporciones entre canales
    return E / np.maximum(1, E.max(axis=-1, keepdims=True))

def luminancia(E):
    # Coeficientes utilizados en la capsula
    return 0.177 * E[..., 0] + 0.813 * E[..., 1] + 0.011 * E[..., 2]


def reescalar(g, p_bajo=0.5, p_alto=99.5):
    lo, hi = np.percentile(g, [p_bajo, p_alto])
    return np.clip((g - lo) / max(hi - lo, EPS), 0, 1)

EPS = 1e-12
def compresion_log(Y):
    logY = np.log(Y + EPS)
    return reescalar(logY, 0, 100)



# filtro homomórfico:
def H_homomorfico(forma, gamma_H, gamma_L, D0, c=1):
    P, Q = forma

    v, u = np.mgrid[0:P, 0:Q]
    D2 = (u - Q / 2)**2 + (v - P / 2)**2

    return gamma_L + (gamma_H - gamma_L) * (1 - np.exp(-c * D2 / D0**2))


def filtro_homomorfico(Y, gamma_H, gamma_L, D0, c=1):
    return


# metodo de durand:

