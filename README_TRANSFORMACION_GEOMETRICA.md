# Transformaciones Geométricas en Imágenes - Guía Completa

Este documento explica línea por línea el código del notebook de transformaciones geométricas y proporciona guías para ajustar los parámetros según diferentes casos de uso.

## 📋 Tabla de Contenidos
1. [Importación de Bibliotecas](#importación-de-bibliotecas)
2. [Carga de Imagen](#carga-de-imagen)
3. [Función de Transformación Geométrica](#función-de-transformación-geométrica)
4. [Tipos de Transformaciones](#tipos-de-transformaciones)
5. [Guía de Parámetros](#guía-de-parámetros)
6. [Caso Especial: Enderezar Reloj Inclinado](#caso-especial-enderezar-reloj-inclinado)

---

## Importación de Bibliotecas

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

**Explicación línea por línea:**
- `import numpy as np`: Importa NumPy para operaciones matriciales y cálculos numéricos
- `import cv2`: Importa OpenCV para procesamiento de imágenes
- `import matplotlib.pyplot as plt`: Importa Matplotlib para visualización de imágenes

---

## Carga de Imagen

```python
!wget https://res.cloudinary.com/dnz4gqdqw/image/upload/v1759856432/louvre.png
```
- Descarga la imagen desde Cloudinary usando el comando wget

```python
X = cv2.imread('louvre.png',0)
```
- `cv2.imread('louvre.png',0)`: Lee la imagen en escala de grises (el parámetro `0` indica escala de grises)
- `X`: Almacena la imagen como una matriz NumPy

```python
print('size = ',X.shape)
```
- Imprime las dimensiones de la imagen (filas, columnas)

```python
plt.figure(figsize=(10,10))
```
- Crea una figura de 10x10 pulgadas para visualización

```python
plt.imshow(X,cmap='gray')
```
- `plt.imshow()`: Muestra la imagen
- `cmap='gray'`: Usa mapa de colores en escala de grises

```python
plt.show()
```
- Renderiza y muestra la figura

---

## Función de Transformación Geométrica

```python
def geo_transformation(X, A, Yshape=None):
```
- Define la función que aplica transformaciones geométricas
- `X`: Imagen de entrada
- `A`: Matriz de transformación afín (2x3)
- `Yshape`: Dimensiones opcionales de la imagen de salida

```python
if Yshape is None:
    (N,M) = X.shape
else:
    (N,M) = Yshape
```
- Si no se especifica `Yshape`, usa las dimensiones de la imagen original
- `N`: Número de filas, `M`: Número de columnas

```python
Y = np.zeros(((N,M)),np.uint8)
```
- Crea una matriz de ceros (imagen vacía) del mismo tamaño que la salida
- `np.uint8`: Tipo de dato entero sin signo de 8 bits (valores 0-255)

```python
m = np.ones((N*M,3))
```
- Crea una matriz de coordenadas homogéneas (i, j, 1) para cada píxel
- Tamaño: (N*M filas) x (3 columnas)

```python
t = 0
for i in range(N):
    for j in range(M):
        m[t,0:3] = [i,j,1]
        t = t+1
```
- Llena la matriz `m` con las coordenadas de cada píxel
- `[i,j,1]`: Coordenadas homogéneas (fila, columna, 1)
- `t`: Contador que recorre cada píxel linealmente

```python
m0 = np.dot(A,m.T)
```
- Aplica la transformación afín multiplicando la matriz A por las coordenadas
- `m.T`: Transpuesta de m (3 x N*M)
- Resultado: Nuevas coordenadas transformadas (i', j')

```python
mpf = np.fix(m0).astype(int)
```
- `np.fix()`: Trunca a enteros (redondea hacia cero)
- `astype(int)`: Convierte a tipo entero
- Esto implementa la interpolación por truncamiento

```python
i0 = mpf[0,:]
j0 = mpf[1,:]
```
- Extrae las coordenadas i' y j' transformadas

```python
kti = np.logical_and(i0>=0,i0<N)
ktj = np.logical_and(j0>=0,j0<M)
kt = np.logical_and(kti,ktj)
```
- `kti`: Verifica que las filas transformadas estén dentro del rango [0, N)
- `ktj`: Verifica que las columnas transformadas estén dentro del rango [0, M)
- `kt`: Combina ambas condiciones (solo píxeles válidos)

```python
t = 0
for i in range(N):
    for j in range(M):
        if kt[t]:
            Y[i,j] = X[i0[t],j0[t]]
        t = t+1
```
- Recorre cada píxel de la imagen de salida
- Si las coordenadas transformadas son válidas (`kt[t]`), copia el valor del píxel de la imagen original
- Realiza el mapeo inverso: (i,j) en Y ← (i0,j0) en X

```python
return Y
```
- Retorna la imagen transformada

---

## Tipos de Transformaciones

### 1. TRASLACIÓN

```python
a11 = 1
a12 = 0
a13 = -100
```
- `a11 = 1`: Sin cambio en escala horizontal
- `a12 = 0`: Sin rotación/sesgo
- `a13 = -100`: Desplazamiento vertical de -100 píxeles (hacia arriba)

```python
a21 = 0
a22 = 1
a23 = -250
```
- `a21 = 0`: Sin rotación/sesgo
- `a22 = 1`: Sin cambio en escala vertical
- `a23 = -250`: Desplazamiento horizontal de -250 píxeles (hacia la izquierda)

```python
a1 = np.array(([a11,a12,a13]))
a2 = np.array(([a21,a22,a23]))
A = np.vstack(([a1,a2]))
```
- Crea la matriz de transformación A = [[a11, a12, a13], [a21, a22, a23]]
- `np.vstack()`: Apila verticalmente los arrays

**Matriz de Traslación:**
```
A = [ 1   0  -100 ]
    [ 0   1  -250 ]
```

### 2. ROTACIÓN

```python
theta = 45.0 / 180.0 * np.pi
```
- Define el ángulo de rotación: 45 grados convertidos a radianes
- Fórmula: radianes = grados * π / 180

```python
a11 = np.cos(theta)
a12 = np.sin(theta)
a13 = -200
a21 = -np.sin(theta)
a22 = np.cos(theta)
a23 = 400
```
- `a11, a12, a21, a22`: Forman la matriz de rotación estándar
- `a13, a23`: Traslación adicional para reposicionar la imagen rotada

**Matriz de Rotación (sin traslación):**
```
R = [ cos(θ)   sin(θ)   0 ]
    [-sin(θ)   cos(θ)   0 ]
```

### 3. ESCALA + ROTACIÓN

```python
theta = 45.0 / 180.0 * np.pi
s = 0.7
```
- `theta`: Ángulo de rotación
- `s = 0.7`: Factor de escala (70% del tamaño original)

```python
a11 = s*np.cos(theta)
a12 = s*np.sin(theta)
a13 = -200

a21 = -s*np.sin(theta)
a22 = s*np.cos(theta)
a23 = 400
```
- Multiplica la matriz de rotación por el factor de escala `s`
- Combina rotación y escalado en una sola transformación

**Matriz de Escala + Rotación:**
```
A = [ s*cos(θ)   s*sin(θ)  -200 ]
    [-s*sin(θ)   s*cos(θ)   400 ]
```

---

## Guía de Parámetros

### 🔧 ¿Qué valores ajustar para diferentes imágenes?

#### **Para TRASLACIÓN:**
- **a13**: Controla el desplazamiento VERTICAL
  - Valores negativos: mueve hacia ARRIBA
  - Valores positivos: mueve hacia ABAJO
- **a23**: Controla el desplazamiento HORIZONTAL
  - Valores negativos: mueve hacia la IZQUIERDA
  - Valores positivos: mueve hacia la DERECHA

**Ejemplo práctico:**
```python
# Mover imagen 50px a la derecha y 30px abajo
a11 = 1; a12 = 0; a13 = 30
a21 = 0; a22 = 1; a23 = 50
```

#### **Para ROTACIÓN:**
- **theta**: Ángulo de rotación
  - Valores positivos: rotación anti-horaria
  - Valores negativos: rotación horaria
  - Común: 0°, 45°, 90°, 180°, 270°
- **a13, a23**: Ajustar según el centro de rotación deseado

**Ejemplo práctico:**
```python
# Rotar 30 grados en sentido horario
theta = -30.0 / 180.0 * np.pi
a11 = np.cos(theta); a12 = np.sin(theta); a13 = -100
a21 = -np.sin(theta); a22 = np.cos(theta); a23 = 200
```

#### **Para ESCALA:**
- **s**: Factor de escala
  - `s > 1`: Amplía la imagen
  - `s = 1`: Tamaño original
  - `0 < s < 1`: Reduce la imagen
  - Ejemplo: `s = 0.5` = 50% del tamaño, `s = 2.0` = 200% del tamaño

**Ejemplo práctico:**
```python
# Ampliar imagen al 150% con rotación de 15°
theta = 15.0 / 180.0 * np.pi
s = 1.5
a11 = s*np.cos(theta); a12 = s*np.sin(theta); a13 = -150
a21 = -s*np.sin(theta); a22 = s*np.cos(theta); a23 = 300
```

---

## Caso Especial: Enderezar Reloj Inclinado

### 🕐 Problema: Reloj inclinado con forma de óvalo

Para **enderezar un reloj inclinado con forma ovalada**, necesitas aplicar dos transformaciones:

1. **ROTACIÓN**: Para corregir la inclinación
2. **CORRECCIÓN DE PERSPECTIVA/SHEAR**: Para corregir la deformación ovalada

### Paso 1: Medir la inclinación

Primero, determina visualmente el ángulo de inclinación del reloj. Por ejemplo, si está inclinado aproximadamente 20° a la derecha, necesitas rotarlo -20°.

### Paso 2: Código para enderezar (solo rotación)

```python
# Si el reloj está inclinado 20° hacia la derecha
theta = -20.0 / 180.0 * np.pi  # Negativo para rotar en sentido horario

a11 = np.cos(theta)
a12 = np.sin(theta)
a13 = 0  # Ajustar según necesidad de centrado

a21 = -np.sin(theta)
a22 = np.cos(theta)
a23 = 0  # Ajustar según necesidad de centrado

a1 = np.array([a11, a12, a13])
a2 = np.array([a21, a22, a23])
A = np.vstack([a1, a2])

Y = geo_transformation(X, A)
plt.figure(figsize=(10,10))
plt.imshow(Y, cmap='gray')
plt.show()
```

### Paso 3: Corregir la forma ovalada (Transformación Afín completa)

Si después de rotar el reloj sigue viéndose como un óvalo en lugar de un círculo, necesitas una **transformación afín más compleja** que incluya shear (cizallamiento):

```python
# Parámetros para corregir óvalo
theta = -20.0 / 180.0 * np.pi  # Ángulo de inclinación
sx = 1.0    # Factor de escala horizontal (ajustar entre 0.8 - 1.2)
sy = 1.2    # Factor de escala vertical (ajustar para corregir óvalo)
shear_x = 0.0  # Cizallamiento horizontal (probar valores entre -0.3 y 0.3)

a11 = sx * np.cos(theta) + shear_x * np.sin(theta)
a12 = sx * np.sin(theta)
a13 = -100  # Ajustar centrado vertical

a21 = -sy * np.sin(theta) + shear_x * np.cos(theta)
a22 = sy * np.cos(theta)
a23 = 200  # Ajustar centrado horizontal

a1 = np.array([a11, a12, a13])
a2 = np.array([a21, a22, a23])
A = np.vstack([a1, a2])

Y = geo_transformation(X, A)
plt.figure(figsize=(10,10))
plt.imshow(Y, cmap='gray')
plt.show()
```

### 📊 Tabla de Ajuste de Parámetros para Reloj Ovalado

| Parámetro | Qué hace | Valores a probar | Efecto en reloj ovalado |
|-----------|----------|------------------|-------------------------|
| `theta` | Corrige inclinación | -45° a 45° | Endereza el reloj |
| `sx` | Escala horizontal | 0.8 - 1.2 | Corrige ancho del óvalo |
| `sy` | Escala vertical | 0.8 - 1.2 | Corrige alto del óvalo |
| `shear_x` | Cizallamiento horizontal | -0.3 a 0.3 | Corrige sesgo/perspectiva |
| `a13` | Centrado vertical | -500 a 500 | Posiciona la imagen |
| `a23` | Centrado horizontal | -500 a 500 | Posiciona la imagen |

### 🎯 Estrategia de Ajuste Paso a Paso

1. **Primero**: Ajusta `theta` hasta que el reloj esté derecho
2. **Segundo**: Si se ve ovalado verticalmente, ajusta `sy` (aumenta si está achatado, reduce si está alargado)
3. **Tercero**: Si se ve ovalado horizontalmente, ajusta `sx`
4. **Cuarto**: Si hay perspectiva/sesgo, ajusta `shear_x` en pequeños incrementos (±0.1)
5. **Finalmente**: Ajusta `a13` y `a23` para centrar la imagen

### Ejemplo Completo para Reloj Específico

```python
# Reloj inclinado 15° a la derecha, ovalado verticalmente
X = cv2.imread('relojdelado.png', 0)

theta = -15.0 / 180.0 * np.pi  # Corregir inclinación
sx = 1.0      # Sin cambio horizontal
sy = 1.15     # Estirar 15% verticalmente para corregir óvalo
shear_x = 0.05  # Pequeño ajuste de perspectiva

a11 = sx * np.cos(theta) + shear_x * np.sin(theta)
a12 = sx * np.sin(theta)
a13 = -50

a21 = -sy * np.sin(theta) + shear_x * np.cos(theta)
a22 = sy * np.cos(theta)
a23 = 100

a1 = np.array([a11, a12, a13])
a2 = np.array([a21, a22, a23])
A = np.vstack([a1, a2])

Y = geo_transformation(X, A)
plt.figure(figsize=(10,10))
plt.imshow(Y, cmap='gray')
plt.title('Reloj Corregido')
plt.show()
```

---

## 💡 Consejos Prácticos

1. **Experimenta iterativamente**: Cambia un parámetro a la vez y observa el resultado
2. **Usa valores pequeños al inicio**: Comienza con cambios pequeños (±5°, ±0.1 en escala)
3. **Guarda configuraciones que funcionen**: Anota los valores que dan buenos resultados
4. **Para imágenes diferentes**: Necesitarás ajustar principalmente `a13` y `a23` para centrado
5. **Interpolación**: Esta implementación usa truncamiento; para mejor calidad, considera implementar interpolación bilineal

---

## 🔬 Fórmulas Matemáticas

### Transformación Afín General:
```
[i']   [a11  a12  a13]   [i]
[j'] = [a21  a22  a23] × [j]
[1 ]   [ 0    0    1 ]   [1]
```

### Componentes:
- **Rotación**: a11=cos(θ), a12=sin(θ), a21=-sin(θ), a22=cos(θ)
- **Escala**: Multiplica componentes por sx, sy
- **Traslación**: a13 (vertical), a23 (horizontal)
- **Shear**: Agrega términos mixtos

---

## 📝 Resumen

Este notebook implementa transformaciones geométricas básicas usando matrices afines. La clave está en entender cómo cada componente de la matriz A afecta la imagen:

- **Diagonal (a11, a22)**: Escala y rotación
- **Off-diagonal (a12, a21)**: Rotación y shear
- **Última columna (a13, a23)**: Traslación

Para cualquier imagen nueva, ajusta estos parámetros según tus necesidades específicas siguiendo las guías proporcionadas.
