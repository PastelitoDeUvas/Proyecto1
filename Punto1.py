
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Lectura de los datos 2D
data_2d = pd.read_csv("data_2d.csv")

# Mostrar los primeros 5 registros
print("Datos 2D:")
print(data_2d.head())


# Lectura de los datos 3D
data_3d = pd.read_csv("data_3d.csv")

# Mostrar los primeros 5 registros
print("Datos 3D:")
print(data_3d.head())

# Estudio estadistico de los datos 2D
print("Estudio estadistico de los datos 2D:")
print(data_2d.describe())

# Estudio estadistico de los datos 3D
print("Estudio estadistico de los datos 3D:")
print(data_3d.describe())

# Gráfica de los datos 2D
plt.figure(figsize=(8, 6))
plt.scatter(data_2d["x"], data_2d["y"], s=50)
plt.title("Datos 2D")
# Gráfica de los datos 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d["x"], data_3d["y"], data_3d["z"], s=50)
ax.set_title("Datos 3D")
plt.show()

################

def initialize_centroids(X, k):
    np.random.seed(412)
    centroids = [X[np.random.randint(len(X))]]  # Primer centroide aleatorio
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        probabilities = distances / distances.sum()
        new_centroid = X[np.random.choice(len(X), p=probabilities)]
        centroids.append(new_centroid)
    return np.array(centroids)

def assign_clusters(X, centroids):
    distances = np.array([[np.linalg.norm(x - centroid) for centroid in centroids] for x in X])
    return np.argmin(distances, axis=1)

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
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        inertia += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
    return inertia

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    print("Centroides iniciales:", centroids)
    
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        if np.mean(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            print("Convergencia alcanzada.")
            break
        
        centroids = new_centroids
    
    inertia = compute_inertia(X, centroids, labels)
    print(f"Inercia final: {inertia}")
    print("Centroides finales:", centroids)
    return centroids, labels, inertia

# Cargar y normalizar datos
data_2d = pd.read_csv("data_2d.csv").values
data_3d = pd.read_csv("data_3d.csv").values
scaler = StandardScaler()
data_2d = scaler.fit_transform(data_2d)
data_3d = scaler.fit_transform(data_3d)

# Número de clusters
k = 5

# Aplicar k-means a data_2d
centroids_2d, labels_2d, inertia_2d = kmeans(data_2d, k)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_2d, cmap='viridis', alpha=0.6)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100)
plt.title(f"K-Means Clustering - 2D Data\nInercia: {inertia_2d:.2f}")
plt.show()

# Aplicar k-means a data_3d
centroids_3d, labels_3d, inertia_3d = kmeans(data_3d, k)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels_3d, cmap='viridis', alpha=0.6)
ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='x', s=100)
ax.set_title(f"K-Means Clustering - 3D Data\nInercia: {inertia_3d:.2f}")
plt.show()


