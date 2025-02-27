#librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Funciones para K-Means con distintas métricas de distancia
def euclidean_distance(A, B):
    return np.linalg.norm(A - B, axis=1)


# Leer el archivo CSV
mall_data = pd.read_csv('Mall_Customers.csv')
# Mostrar las primeras filas del DataFrame
print("Datos de Mall Customers:")
print(mall_data.head())


# Eliminar las variables categóricas
mall_data_numeric = mall_data.select_dtypes(include=[np.number])
mall_data_numeric = mall_data_numeric.drop(columns=['CustomerID'])
# Centralizar las variables restantes
mall_data_centered = mall_data_numeric - mall_data_numeric.mean()

# Dividir por la desviación estándar de cada columna
mall_data_standardized = mall_data_centered / mall_data_centered.std()

# Mostrar las primeras filas del DataFrame estandarizado
print("Datos estandarizados de Mall Customers:")
print(mall_data_standardized.head())


# Estudio estadistico 
print("Estudio estadistico:")
print(mall_data_standardized.describe())



# Gráfica de los datos 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mall_data_standardized["Age"], mall_data_standardized["Annual Income (k$)"],mall_data_standardized["Spending Score (1-100)"], s=50)
ax.set_title("Datos ")
plt.show()

scaler = StandardScaler()
mall_data_standardized = scaler.fit_transform(mall_data_standardized)
########################################################################################################################################

def initialize_centroids(X, k):
    np.random.seed(41342)
    centroids = [X[np.random.randint(len(X))]]  # Primer centroide aleatorio
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        probabilities = distances / distances.sum()
        new_centroid = X[np.random.choice(len(X), p=probabilities)]
        centroids.append(new_centroid)
    return np.array(centroids)

def assign_clusters(X, centroids, metric_func):
    distances = np.array([metric_func(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            farthest_point = X[np.argmax([np.min([np.linalg.norm(x - c) for c in new_centroids]) for x in X])]
            new_centroids.append(farthest_point)
    return np.array(new_centroids)

def compute_inertia(X, centroids, labels):
    inertia = 0
    labels = np.array(labels)
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

def kmeans(X, k, metric_func, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids, metric_func)
        new_centroids = update_centroids(X, labels, k)
        if np.mean(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            break
        centroids = new_centroids
    inertia = compute_inertia(X, centroids, labels)
    return centroids, labels, inertia

#######################################################################################################################


centroids_3d, labels_3d, inertia_3d = kmeans(mall_data_standardized, 3, euclidean_distance)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mall_data_standardized[:, 0], mall_data_standardized[:, 1], mall_data_standardized[:, 2], c=labels_3d, cmap='viridis', alpha=0.6)
ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='x', s=100)
ax.set_title(f"K-Means Euclidiano\nInercia: {inertia_3d:.2f}")
plt.show()




