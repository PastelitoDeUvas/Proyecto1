import numpy as np

def modelo(x, params):
    """
    Define el modelo no lineal que queremos ajustar.
    En este caso, usamos una función exponencial como ejemplo.
    """
    a, b = params
    return a * np.exp(b * x)

def jacobiano(x, params):
    """
    Calcula la matriz Jacobiana (derivadas parciales del modelo respecto a los parámetros).
    """
    a, b = params
    J = np.zeros((len(x), len(params)))
    J[:, 0] = np.exp(b * x)  # Derivada respecto a 'a'
    J[:, 1] = a * x * np.exp(b * x)  # Derivada respecto a 'b'
    return J

def newton_gauss(x, y, params_iniciales, tol=1e-6, max_iter=100):
    """
    Implementación del método de Newton-Gauss para ajuste de regresión no lineal.
    """
    params = np.array(params_iniciales, dtype=float)
    
    for _ in range(max_iter):
        y_pred = modelo(x, params)

        residuo = y - y_pred
        J = jacobiano(x, params)
        
        # Calcular el ajuste usando la pseudo-inversa de Moore-Penrose
        delta = np.linalg.pinv(J) @ residuo
        
        params += delta
        
        # Criterio de convergencia
        if np.linalg.norm(delta) < tol:
            break
    
    return params

# Datos de ejemplo (x, y)
x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = np.array([2.0, 2.7, 4.0, 5.9, 8.5, 12.2], dtype=float)  # Valores ruidosos de una exponencial

# Parámetros iniciales (adivinados)
params_iniciales = [1.0, 0.5]

# Ejecutar el algoritmo
params_finales = newton_gauss(x, y, params_iniciales)
#print("Parámetros ajustados:", params_finales)
print(modelo(x,params_iniciales))