import numpy as np
from kmedoids import KMedoids
import matplotlib.pyplot as plt

# Define the dataset
data_points = np.array([(1, 2), (2, 3), (5, 8), (6, 7), (8, 9), (10, 12), (11, 13), (14, 15), (16, 18)])

# Perform k-medoids clustering
k = 2
kmedoids_instance = KMedoids(n_clusters=k, random_state=0).fit(data_points)
clusters = kmedoids_instance.labels_
medoids = kmedoids_instance.medoid_indices_

# Assign colors for clusters
colors = ['b', 'g']

# Function to plot clusters with circles
def plot_clusters(data_points, medoids, clusters):
    plt.figure(figsize=(8, 6))
    for cluster_index in range(k):
        cluster_color = colors[cluster_index]
        cluster_points = data_points[clusters == cluster_index]
        medoid = data_points[medoids[cluster_index]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color, label=f'Cluster {cluster_index}', marker='o', edgecolor='k')
        plt.scatter(medoid[0], medoid[1], color=cluster_color, marker='s', s=100, edgecolor='k')
        # Compute radius of circle
        radius = max([np.linalg.norm(point - medoid) for point in cluster_points])
        circle = plt.Circle(medoid, radius, color=cluster_color, alpha=0.1)
        plt.gca().add_artist(circle)
    plt.title('K-medoids Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot clusters with circles
plot_clusters(data_points, medoids, clusters)
