import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd


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

def r2_matriz(matriz):
    matriz = np.atleast_2d(matriz)  # Asegurar que sea bidimensional
    m, n = matriz.shape  # Obtener dimensiones correctamente

    columnas = [np.ones(m)]  # Primera columna de 1s (intercepto en regresión)

    for g in range(1, 3):
        for i in range(n):  
            columnas.append(matriz[:, i]**g)  # Agregar términos elevados a `g`

    return np.column_stack(columnas)

def r2(a, b, beta):
    a = r2_matriz(a) 
    a = np.atleast_2d(a)  # Asegurar que `a` es bidimensional
    beta = beta.reshape(-1, 1) if beta.ndim == 1 else beta  # Convertir `beta` a (n,1)
    b = b.reshape(-1, 1) if b.ndim == 1 else b  # Asegurar que `b` sea (m,1)
    aprox = a @ beta  # Multiplicación segura

    ss_res = np.sum((b - aprox) ** 2)  # Suma de los residuos al cuadrado
    ss_tot = np.sum((b - np.mean(b)) ** 2)  # Suma total de cuadrados
    
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

def error(a, b, beta):
    

    aprox = a @ beta  # Multiplicación segura

    
    error = np.mean((b - aprox) ** 2) 
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

    for k in range(1, 10):
        errores = []  # Lista para los errores
        grados = []  # Lista de grados
        betas_list = []  # Lista para los betas

        a_20, b_20, a_80, b_80 = split_matrices(a.T, b, seed=k**7)

        for i in range(1, 10):  
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

    
    best_error, i, j = encontrar_minimo(all_errores)
    best_beta = all_betas[i][j]  # Indexación corregida
    best_grado =all_grados[i][j]
    

    return best_error, best_beta,best_grado

def gradiente_training(a, b):
    all_errores = []  # Lista para los errores
    all_betas = []  # Lista para los betas
    all_grados =[]
    all_iteraciones=[]

    plt.figure(figsize=(8, 5))  # Crear una sola figura

    for k in range(1, 10):
        errores = []  # Lista para los errores
        grados = []  # Lista de grados
        betas_list = []  # Lista para los betas
        iteraciones_list=[]

        a_20, b_20, a_80, b_80 = split_matrices(a.T, b.T, seed=k**7)

        for i in range(1, 10):  
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

    
    best_error, i, j = encontrar_minimo(all_errores)
    best_beta = all_betas[i][j]  # Indexación corregida
    best_grado =all_grados[i][j]
    iteration =all_iteraciones[i][j]
    
    return best_error, best_beta,best_grado,iteration

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



    data = pd.read_csv("data.csv")


def cleaning_data(data):
    # Convertir la columna 'genero' a binario: suponiendo que 'Femenino' y 'Masculino' son los valores
    data["genero"] = data["genero"].map({"F": 0, "M": 1})

    # Eliminar columnas no útiles (como ID y target)
    X = data.drop(columns=[ "target"])

    X = X.apply(pd.to_numeric, errors="coerce")
    


    # Eliminar filas con valores faltantes (opcional, según si querés trabajar con NaNs o no)
    X = X.dropna(axis=1, thresh=int(len(X) * 0.9))  # Conserva columnas con al menos 90% de datos válidos


    # Extraer la variable objetivo alineada con X
    y = data.loc[X.index, "target"].values.reshape(-1, 1)


    # Copiar columnas que no se van a normalizar
    X_no_norm = X[["genero", "historial_diabetes"]]


    # Columnas que sí se normalizan
    X_to_norm = X.drop(columns=[ "genero","historial_diabetes"])

    means = X_to_norm.mean()
    stds = X_to_norm.std()
    stds[stds == 0] = 1  # Evitar división por cero

    X_normalized = (X_to_norm - means) / stds

    # Combinar todo de nuevo (ordenando columnas si querés)
    X_final = pd.concat([X_normalized, X_no_norm], axis=1)

    # Opcional: asegurarse que las columnas queden en el mismo orden original
    X = X_final[X.columns]  # conserva el orden original


    return X,y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    for epoch in range(epochs):
        # Predicciones
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # Cálculo de pérdida (binary cross-entropy)
        loss = -np.mean(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8))

        # Gradientes
        dW = np.dot(X.T, (A - y)) / m
        db = np.mean(A - y)

        # Actualización de pesos
        W -= lr * dW
        b -= lr * db

        # Imprimir la pérdida cada 100 iteraciones
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W, b


def predict(X, W, b, threshold=0.5):
    probs = sigmoid(np.dot(X, W) + b)
    return (probs >= threshold).astype(int)

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0  # Para evitar división por cero

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

def find_best_threshold(y_true, probs):
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.01, 0.9, 0.01):
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def logistic_regression_with_regularization(X, y, lr=0.01, epochs=1000, lambda_reg=0.1):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    for epoch in range(epochs):
        # Predicciones
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # Cálculo de pérdida (con regularización L2)
        loss = -np.mean(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8)) + lambda_reg * np.sum(W**2)

        # Gradientes
        dW = np.dot(X.T, (A - y)) / m + 2 * lambda_reg * W  # Regularización en los gradientes
        db = np.mean(A - y)

        # Actualización de pesos
        W -= lr * dW
        b -= lr * db

        # Imprimir la pérdida cada 100 iteraciones
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W, b


def new_cleaning_data(data):
    # Convertir la columna 'genero' a binario: suponiendo que 'Femenino' y 'Masculino' son los valores
    data["genero"] = data["genero"].map({"F": 0, "M": 1})

    # Eliminar columnas no útiles (como ID y target)
    X = data.drop(columns=[ "target","ratio_colesterol","actividad_fisica","horas_sueno","historial_diabetes","dias_ultima_consulta","edad","genero"])


    X = X.apply(pd.to_numeric, errors="coerce")
    


    # Eliminar filas con valores faltantes (opcional, según si querés trabajar con NaNs o no)
    X = X.dropna(axis=1, thresh=int(len(X) * 0.9))  # Conserva columnas con al menos 90% de datos válidos


    # Extraer la variable objetivo alineada con X
    y = data.loc[X.index, "target"].values.reshape(-1, 1)





    # Columnas que sí se normalizan
    X_to_norm = X

    means = X_to_norm.mean()
    stds = X_to_norm.std()
    stds[stds == 0] = 1  # Evitar división por cero

    X_normalized = (X_to_norm - means) / stds

    # Combinar todo de nuevo (ordenando columnas si querés)
    X_final = X_normalized

    # Opcional: asegurarse que las columnas queden en el mismo orden original
    X = X_final[X.columns]  # conserva el orden original


    return X,y