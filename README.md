# Tarea 4 — IEE2714 Fundamentos de Procesamiento de Imágenes (2° 2026)

Este es el repo recuperado ya que ubieron bugs al meter las imagenes al repo anterior, no supe solucionarlo asi que cree otro repo que no tiene los comits iniciales. Lo siento!



## Estructura del repositorio

```
Tarea_4_Imagenes/
├── archivos_t4/              # imágenes HDR (Estan en un gitignore)
├── Pregunta 1/
│   ├── Implementacion.py     # pirámide Laplaciana, starlet y reconstrucción
│   ├── Experimentacion.py    # experimentos 1, 2 y 3
│   └── Resultados/           # figuras generadas
└── Pregunta 2/
    └── Implementacion.py     # funciones de compresión HDR y recuperación de color
```

## Requisitos

Python 3 con:

```
pip install numpy scipy matplotlib tifffile scikit-image opencv-python
```

## Imágenes

Las imágenes HDR no se incluyen en el repositorio (la carpeta `archivos_t4/` está en el `.gitignore`).
Para reproducir los resultados, crear la carpeta `archivos_t4/` en la raíz del repo y copiar ahí los archivos `.tif`.
La Pregunta 1 usa `memorial.tif` por defecto; se puede indicar otra imagen pasando su ruta como argumento.

## Pregunta 1

Los scripts se pueden ejecutar desde cualquier carpeta, ya que las rutas se calculan respecto a cada archivo.
Todas las figuras se guardan en `Pregunta 1/Resultados/`.

**Implementación y verificación de reconstrucción**

```
python "Pregunta 1/Implementacion.py" [ruta_imagen.tif]
```

Carga la luminancia de la imagen, le aplica una compresión global (Reinhard, clave = 0.18) y la normaliza a [0, 1].
Luego descompone con la pirámide Laplaciana y con starlet (5 capas de detalle + residual), reconstruye e imprime el
RMSE de reconstrucción de ambas. Guarda `luminancia_comprimida.png`.

**Experimentación**

```
python "Pregunta 1/Experimentacion.py" [ruta_imagen.tif]
```

Los experimentos se eligen en el bloque `if __name__ == "__main__":` al final del archivo, comentando los que no se
quieren ejecutar y descomentando los que sí (`experimento_1()`, `experimento_2()`, `experimento_3()`).

| Experimento | Qué hace | Archivos generados |
|---|---|---|
| 1 | Capas de detalle y residual de ambas representaciones, con sus dimensiones | `p1_exp1_capas.png` |
| 2 | Umbral duro `T_j = α·max\|d_j\|` con α ∈ {0.05, 0.20} (se cambian en la variable `alphas`). Imprime en consola el % de coeficientes conservados por capa y el RMSE | `p1_exp2_umbral_globales.png`, `p1_exp2_umbral_recortes.png` |
| 3 | Ganancias por escala: `g_fino = [2.5, 1.8, 1.2, 1.0, 1.0]` y `g_grueso = [1.0, 1.0, 1.2, 1.8, 2.5]`, con dos recortes ampliados | `p1_exp3_ganancias_globales.png`, `p1_exp3_ganancias_recorte1.png`, `p1_exp3_ganancias_recorte2.png` |

## Pregunta 2

Por ahora el repositorio incluye solo las funciones de `Pregunta 2/Implementacion.py` (sin script de ejecución ni figuras):

- `cargar_rgb_hdr(ruta)`: lee el `.tif` HDR y devuelve los 3 canales RGB sin valores negativos.
- `luminancia(E)`: luminancia a partir de RGB.
- `filtro_homomorfico(Y, gamma_H=1.5, gamma_L=0.5, D0=30, c=1.0)`: filtro homomórfico sobre la luminancia.
- `durand(Y, sigma_s=None, sigma_r=0.4, contraste=30)`: separación base/detalle con filtro bilateral (`sigma_s` por defecto = 0.02 · lado mayor).
- `recolorear(E, Y, Y_out, beta=1)`: reconstrucción de color por razón de luminancia, `C' = Y'·(C/(Y+ε))^β`.
- `recolorear_hsv(E, Y, Y_out, k_s=1)`: reconstrucción con `β = 1` y ajuste posterior de la saturación en HSV (`S' = clip(k_s·S, 0, 1)`).
