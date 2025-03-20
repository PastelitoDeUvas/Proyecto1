import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Función para calcular la distancia euclidiana
def euclidean_distance(A, B):
    """
    Calcula la distancia euclidiana entre dos conjuntos de puntos.
    
    Parámetros:
    A: np.array, conjunto de puntos de referencia.
    B: np.array, punto o conjunto de puntos a medir.
    
    Retorna:
    np.array con las distancias euclidianas calculadas.
    """
    return np.linalg.norm(A - B, axis=1)

# Cargar los datos desde el archivo CSV
mall_data = pd.read_csv('Mall_Customers.csv')
print("Datos de Mall Customers:")
print(mall_data.head())

# Seleccionar solo las columnas numéricas y eliminar 'CustomerID'
mall_data_numeric = mall_data.select_dtypes(include=[np.number]).drop(columns=['CustomerID'])

# Centralizar las variables restando la media de cada una
mall_data_centered = mall_data_numeric - mall_data_numeric.mean()

# Estandarizar dividiendo por la desviación estándar
mall_data_standardized = mall_data_centered / mall_data_centered.std()
print("Datos estandarizados de Mall Customers:")
print(mall_data_standardized.head())

# Estudio estadístico de los datos
print("Estudio estadístico:")
print(mall_data_standardized.describe())

# Gráfica de los datos en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mall_data_standardized["Age"], 
           mall_data_standardized["Annual Income (k$)"], 
           mall_data_standardized["Spending Score (1-100)"], 
           s=50)
ax.set_title("Datos")
plt.show()

# Aplicar escalado estándar a los datos
scaler = StandardScaler()
mall_data_standardized = scaler.fit_transform(mall_data_standardized)

# Función para inicializar centroides usando el método K-Means++
def initialize_centroids(X, k):
    """
    Inicializa k centroides utilizando el método K-Means++.
    
    Parámetros:
    X: np.array, datos de entrada.
    k: int, número de centroides a inicializar.
    
    Retorna:
    np.array con los centroides inicializados.
    """
    np.random.seed(41342)
    centroids = [X[np.random.randint(len(X))]]  # Primer centroide aleatorio
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        probabilities = distances / distances.sum()
        new_centroid = X[np.random.choice(len(X), p=probabilities)]
        centroids.append(new_centroid)
    return np.array(centroids)

# Función para asignar clusters a los datos
def assign_clusters(X, centroids, metric_func):
    """
    Asigna cada punto de los datos a su centroide más cercano.
    
    Parámetros:
    X: np.array, datos de entrada.
    centroids: np.array, centroides actuales.
    metric_func: función de distancia a utilizar.
    
    Retorna:
    np.array con las etiquetas de los clusters asignados.
    """
    distances = np.array([metric_func(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)

# Función para actualizar los centroides
def update_centroids(X, labels, k):
    """
    Actualiza los centroides calculando el promedio de los puntos asignados a cada uno.
    
    Parámetros:
    X: np.array, datos de entrada.
    labels: np.array, etiquetas de cluster asignadas.
    k: int, número de clusters.
    
    Retorna:
    np.array con los nuevos centroides.
    """
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            farthest_point = X[np.argmax([np.min([np.linalg.norm(x - c) for c in new_centroids]) for x in X])]
            new_centroids.append(farthest_point)
    return np.array(new_centroids)

# Función para calcular la inercia del clustering
def compute_inertia(X, centroids, labels):
    """
    Calcula la inercia del clustering, es decir, la suma de las distancias cuadradas
    de cada punto a su centroide asignado.
    
    Parámetros:
    X: np.array, datos de entrada.
    centroids: np.array, centroides actuales.
    labels: np.array, etiquetas de cluster asignadas.
    
    Retorna:
    float con el valor de la inercia.
    """
    inertia = 0
    labels = np.array(labels)
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

# Algoritmo de K-Means
def kmeans(X, k, metric_func, max_iters=100, tol=1e-4):
    """
    Implementación del algoritmo de K-Means para clustering.
    
    Parámetros:
    X: np.array, datos de entrada.
    k: int, número de clusters.
    metric_func: función de distancia a utilizar.
    max_iters: int, número máximo de iteraciones (por defecto 100).
    tol: float, criterio de convergencia basado en el cambio promedio de los centroides.
    
    Retorna:
    centroids: np.array, centroides finales.
    labels: np.array, etiquetas de cluster asignadas.
    inertia: float, valor de inercia del clustering.
    """
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids, metric_func)
        new_centroids = update_centroids(X, labels, k)
        if np.mean(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            break
        centroids = new_centroids
    inertia = compute_inertia(X, centroids, labels)
    return centroids, labels, inertia

# Aplicar K-Means con 3 clusters
centroids_3d, labels_3d, inertia_3d = kmeans(mall_data_standardized, 3, euclidean_distance)

# Visualización de los clusters
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mall_data_standardized[:, 0], 
           mall_data_standardized[:, 1], 
           mall_data_standardized[:, 2], 
           c=labels_3d, cmap='viridis', alpha=0.6)
ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='x', s=100)
ax.set_title(f"K-Means Euclidiano\nInercia: {inertia_3d:.2f}")
plt.show()


"""
Impacto de la Métrica de Distancia en la Asignación de Clusters con K-Means

Resultados
Se probó el algoritmo K-Means con distintas métricas de distancia (Euclidiana, Manhattan y Chebyshev) y con diferentes semillas de inicialización. No se observaron cambios significativos en la asignación de clusters ni en la inercia final.

Impacto de la Métrica de Distancia
El uso de diferentes métricas no afectó la asignación de clusters. Esto puede deberse a que los datos tienen clusters bien definidos y la normalización minimizó diferencias entre distancias.

Impacto de la Semilla de Inicialización
El cambio de semilla no alteró los resultados, lo que sugiere que el algoritmo es robusto y converge a soluciones similares independientemente de la inicialización.

Análisis de la Calidad del Clustering
Para  con la distancia euclidiana, se obtuvo una inercia de 304. Esto indica que el agrupamiento es razonablemente bueno, aunque no perfecto. Los clusters muestran cierta estructura, pero también se observa dispersión en los datos. Al cambiar la métrica de distancia, la inercia aumentó un poco, pero no lo suficiente como para hacer una diferencia significativa. Esto sugiere que la elección de la métrica no es un factor determinante en este caso, probablemente porque la distribución de los datos no es esférica y K-Means no siempre es el método más adecuado para este tipo de formas.

En general, el clustering obtenido es aceptable, pero no ideal. Se podría mejorar explorando otros métodos como DBSCAN o GMM, que pueden manejar mejor estructuras de datos más complejas.

Conclusión
En este experimento, ni la métrica de distancia ni la semilla de inicialización tuvieron un impacto relevante en la formación de los clusters. La inercia sugiere un agrupamiento razonable, pero con margen de mejora. Dado que la distribución de los datos no es esférica, K-Means puede no ser la mejor opción. Sería interesante probar con diferentes valores de  y evaluar la calidad del clustering con métricas adicionales, como el Índice de Silueta, para obtener una mejor perspectiva de los resultados.

"""



