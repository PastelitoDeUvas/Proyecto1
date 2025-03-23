import numpy as np

def split_matrices(matrix1, matrix2, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Para reproducibilidad
    
    assert matrix1.shape[0] == matrix2.shape[0], "Las matrices deben tener el mismo número de filas."
    
    num_rows = matrix1.shape[0]
    num_selected = int(0.2 * num_rows)
    
    indices = np.random.choice(num_rows, num_selected, replace=False)
    mask = np.ones(num_rows, dtype=bool)
    mask[indices] = False
    
    # Generar las 4 matrices
    matrix1_20 = matrix1[indices, :]
    matrix2_20 = matrix2[indices, :]
    matrix1_80 = matrix1[mask, :]
    matrix2_80 = matrix2[mask, :]
    
    return matrix1_20, matrix2_20, matrix1_80, matrix2_80

def create_variable_matrix(matriz, grado=1):

    m, n = matriz.shape
    columnas = [np.ones(m)]
    for g in range(1, grado + 1):
        columnas.extend([matriz[:, i]**g for i in range(n)])
    return np.column_stack(columnas)

def result_vector(matriz):

    return matriz[:, -1]



# 🎀 Método de la Pseudoinversa 🎀
def pseudoinversa(A, b):
    AtA = A.T @ A              # Calculamos A^T * A
    Atb = A.T @ b              # Calculamos A^T * b
    beta = np.linalg.solve(AtA, Atb)  # Resolvemos el sistema
    return beta



def gradiente_descendente(A, b, lr=0.01, max_iter=1000000):
    m, n = A.shape
    beta = np.zeros(n)  # Inicializamos los coeficientes en cero
    error_anterior = np.inf
    iteraciones = 0
    
    
    
    
    while iteraciones < max_iter:
        gradiente = -2 * A.T @ (b - A @ beta) / m  # Calculamos el gradiente
        beta -= lr * gradiente  # Ajustamos beta
        error_nuevo = np.linalg.norm(b - A @ beta)  # Error actual

        if error_nuevo > error_anterior:  # Condición de parada si el error sube
            break
        
        error_anterior = error_nuevo
        iteraciones += 1
    
    return beta, iteraciones

# 💖 Función para calcular error 💖
def error(A, b, beta):
    predicciones = A @ beta  # Calculamos las predicciones
    error = np.linalg.norm(b - predicciones)  # Error en norma euclidiana
    return error


def pseudoinversa_data (A,b):
    
    beta_pinv = pseudoinversa(A, b)
    error_pinv = error(A, b, beta_pinv)
    print("\n--- ✨ Solución con Pseudoinversa ✨ ---")
    print(f"Intercepto (β₀): {beta_pinv[0]:.2f}")
    for i in range(1, len(beta_pinv)):
        print(f"Coeficiente β{i}: {beta_pinv[i]:.2f}")
    print(f"Error: {error_pinv:.2f}")
    

def gradiente_descendente_data (A,b):

    beta_gd, iteraciones= gradiente_descendente(A, b)
    error_gd = error(A, b, beta_gd)

    # 🌷 Imprimir resultados del Gradiente Descendente 🌷
    print("\n--- ✨ Solución con Gradiente Descendente ✨ ---")
    print(f"Intercepto (β₀): {beta_gd[0]:.2f}")
    for i in range(1, len(beta_gd)):
        print(f"Coeficiente β{i}: {beta_gd[i]:.2f}")
    
    print(f"Error: {error_gd:.2f}")
    print(f"Número de iteraciones: {iteraciones}")


