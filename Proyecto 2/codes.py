import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def standardize(A):

    mean = np.mean(A, axis=0)  # Media de cada columna
    std = np.std(A, axis=0)    # Desviación estándar de cada columna
    
    return (A - mean) / std  # Estandarización por columnas


def split_matrices(matrix1, matrix2, seed=None):
    if seed is not None:
        np.random.seed(seed)  

    matrix1 = np.atleast_2d(matrix1).T  
    matrix2 = np.atleast_2d(matrix2).T  
    
    assert matrix1.shape[0] == matrix2.shape[0], "Las matrices deben tener el mismo número de filas."

    num_rows = matrix1.shape[0]
    num_selected = int(0.2 * num_rows)

    indices = np.random.choice(num_rows, num_selected, replace=False)
    mask = np.ones(num_rows, dtype=bool)
    mask[indices] = False

    return matrix1[indices], matrix2[indices], matrix1[mask], matrix2[mask]

def quitar_ultima_columna(matriz):
    """Elimina la última columna de una matriz dada."""
    if matriz.shape[1] == 1:
        raise ValueError("La matriz solo tiene una columna, no se puede eliminar.")
    
    return matriz[:, :-1]

def result_vector(matriz):

    return matriz[:, -1]

def create_variable_matrix(matriz, grado=1):


    # Obtener las dimensiones correctamente
    dimensiones = matriz.shape
    if len(dimensiones) == 1:
        m = dimensiones[0]
        n = 1  # Unidimensional, se trata como una sola columna
        matriz = matriz.reshape(m, 1)  # Convertir en matriz de tamaño (m,1)
    else:
        m, n = dimensiones

    columnas = [np.ones(m)]  # Primera columna de 1s (intercepto en regresión)

    for g in range(1, grado + 1):
        columnas.extend([matriz[:, i]**g for i in range(n)])

    return np.column_stack(columnas)

def pseudoinversa(A, b):
    """Calcula la solución de mínimos cuadrados usando la pseudoinversa manualmente.
       Si (A^T A) no es invertible, lo indica."""
    
    AtA = A.T @ A  # A^T * A
    Atb = A.T @ b  # A^T * b
    
    # Verificar si AtA es invertible (determinante distinto de 0)
    if np.linalg.det(AtA) == 0:
        print("❌ La matriz (A^T A) no es invertible. No se puede calcular la pseudoinversa.")
        return None
    
    # Resolver el sistema (A^T A) beta = A^T b
    beta = np.linalg.solve(AtA, Atb)
    
    return beta


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # Suma de los residuos al cuadrado
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Suma total de cuadrados
    return 1 - (ss_res / ss_tot)


def gradiente_descendente(A, b, lr=0.01, max_iter=1000000):
    m, n = A.shape
    b = b.reshape(-1, 1)  # Asegurar que b sea una matriz columna
    beta = np.zeros((n, 1))
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
    
def encontrar_minimo(lista_de_listas):
    """Encuentra el valor mínimo en una lista de listas y su posición."""
    min_valor = float('inf')
    min_i, min_j = -1, -1

    for i, fila in enumerate(lista_de_listas):
        for j, valor in enumerate(fila):
            if valor < min_valor:
                min_valor = valor
                min_i, min_j = i, j

    return min_valor, min_i, min_j

def pseudo_training(a, b):
    all_errores = []  # Lista para los errores
    all_betas = []  # Lista para los betas
    all_grados =[]

    plt.figure(figsize=(8, 5))  # Crear una sola figura

    for k in range(1, 4):
        errores = []  # Lista para los errores
        grados = []  # Lista de grados
        betas_list = []  # Lista para los betas

        a_20, b_20, a_80, b_80 = split_matrices(a.T, b, seed=k**7)

        for i in range(1, 4):  
            a_transf = create_variable_matrix(a_20, grado=i)
            beta = pseudoinversa(a_transf, b_20)

            if beta is None:
                continue  # Si no se puede calcular beta, saltamos la iteración

            betas_list.append(beta)

            a_transf = create_variable_matrix(a_80, grado=i)
            err = error(a_transf, b_80, beta)
            errores.append(err)
            grados.append(i)

        all_errores.append(errores)
        all_betas.append(betas_list)
        all_grados.append(grados)

        plt.plot(grados, errores, marker='o', linestyle='-', label=f'Iter {k}')

    plt.xlabel('Grado del Modelo')
    plt.ylabel('Error')
    plt.title('Error vs. Grado del Modelo')
    plt.legend()
    plt.grid()
    plt.show()

    a_transf = create_variable_matrix(a_80, grado=2)
    best_error, i, j = encontrar_minimo(all_errores)
    best_beta = all_betas[i][j]  # Indexación corregida
    best_grado =all_grados[i][j]
    best_r2=r2(b,a_transf @ best_beta)

    return best_error, best_beta,best_grado,best_r2

def gradiente_training(a, b):
    all_errores = []  # Lista para los errores
    all_betas = []  # Lista para los betas
    all_grados =[]
    all_iteraciones=[]

    plt.figure(figsize=(8, 5))  # Crear una sola figura

    for k in range(1, 4):
        errores = []  # Lista para los errores
        grados = []  # Lista de grados
        betas_list = []  # Lista para los betas
        iteraciones_list=[]

        a_20, b_20, a_80, b_80 = split_matrices(a.T, b.T, seed=k**7)

        for i in range(1, 4):  
            a_transf = create_variable_matrix(a_20, grado=i)
            beta,iteraciones = gradiente_descendente(a_transf, b_20, lr=0.01, max_iter=1000000)

            if beta is None:
                continue  # Si no se puede calcular beta, saltamos la iteración

            betas_list.append(beta)
            iteraciones_list.append(iteraciones)

            a_transf = create_variable_matrix(a_80, grado=i)
            error_cuadratico = error(a_transf, b_80, beta)
            errores.append(error_cuadratico)
            
            grados.append(i)

        all_errores.append(errores)
        all_betas.append(betas_list)
        all_grados.append(grados)
        all_iteraciones.append(iteraciones_list)

        plt.plot(grados, errores, marker='o', linestyle='-', label=f'Iter {k}')

    plt.xlabel('Grado del Modelo')
    plt.ylabel('Error')
    plt.title('Error vs. Grado del Modelo')
    plt.legend()
    plt.grid()
    plt.show()

    a_transf = create_variable_matrix(a_20, grado=2)
    best_error, i, j = encontrar_minimo(all_errores)
    best_beta = all_betas[i][j]  # Indexación corregida
    best_grado =all_grados[i][j]
    iteration =all_iteraciones[i][j]
    best_r2=r2(b,a_transf @ best_beta)
    return best_error, best_beta,best_grado,iteration,best_r2

def condicion(A):
    # Valores singulares de la matriz
    valores_singulares = np.linalg.svd(A, compute_uv=False)
    
    # Valor máximo y mínimo
    sigma_max = max(valores_singulares)
    sigma_min = min(valores_singulares)
    
    # Número de condición
    numero_condicion = sigma_max / sigma_min
    
    # Imprimir en tabla bonita
    tabla = [
        ["Valor Singular Máximo", sigma_max],
        ["Valor Singular Mínimo", sigma_min],
        ["Número de Condición", numero_condicion]
    ]
    
    print(tabulate(tabla, headers=["Concepto", "Valor"], tablefmt="fancy_grid"))


