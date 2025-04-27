import random

# Step 1: Read CSV file manually
def load_data(filepath):
    data_points = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        hours_idx = header.index('Hours_Studied')
        exam_idx = header.index('Exam_Score')

        for line in lines[1:]:
            parts = line.strip().split(',')
            try:
                hours = float(parts[hours_idx])
                score = float(parts[exam_idx])
                data_points.append((hours, score))
            except ValueError:
                continue  # Skip invalid rows
    return data_points

# Step 2: Helper Functions
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def initialize_centroids(points, k):
    return random.sample(points, k)

def assign_clusters(points, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_centroids.append((0, 0))
            continue
        x_mean = sum([point[0] for point in cluster]) / len(cluster)
        y_mean = sum([point[1] for point in cluster]) / len(cluster)
        new_centroids.append((x_mean, y_mean))
    return new_centroids

def has_converged(old_centroids, new_centroids, tol=1e-4):
    for old, new in zip(old_centroids, new_centroids):
        if euclidean_distance(old, new) > tol:
            return False
    return True

# NEW: Mean Squared Error (MSE) function
def calculate_mse(clusters, centroids):
    total_error = 0
    total_points = 0
    for idx, cluster in enumerate(clusters):
        centroid = centroids[idx]
        for point in cluster:
            distance = euclidean_distance(point, centroid)
            total_error += distance ** 2
            total_points += 1
    mse = total_error / total_points if total_points > 0 else 0
    return mse

# Step 3: Full K-Means Algorithm
def k_means(points, k=3, max_epochs=100):
    centroids = initialize_centroids(points, k)
    for epoch in range(max_epochs):
        clusters = assign_clusters(points, centroids)
        new_centroids = update_centroids(clusters)
        mse = calculate_mse(clusters, new_centroids)  # <-- Calculate MSE
        print(f"Epoch {epoch+1}: MSE = {mse:.4f}")

        if has_converged(centroids, new_centroids):
            print(f"Converged after {epoch+1} epochs")
            break
        centroids = new_centroids
    return centroids, clusters

# Step 4: Visualization
def plot_clusters(clusters, centroids):
    import matplotlib.pyplot as plt
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Plot each cluster
    for i, cluster in enumerate(clusters):
        x_vals = [point[0] for point in cluster]
        y_vals = [point[1] for point in cluster]
        plt.scatter(x_vals, y_vals, color=colors[i % len(colors)], label=f"Cluster {i+1}")

    # Plot centroids
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color='black', marker='x', s=100)

    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.title('K-Means Clustering (from scratch)')
    plt.legend()
    plt.show()

# ========================
# Run Everything

if __name__ == "__main__":
    file_path = "C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv"  # Change if needed
    data_points = load_data(file_path)

    print(f"Loaded {len(data_points)} data points.")

    k = 3  # Number of clusters
    final_centroids, final_clusters = k_means(data_points, k)

    print("\nFinal Centroids:")
    for i, centroid in enumerate(final_centroids):
        print(f"Cluster {i+1}: {centroid}")

    print("\nCluster Sizes:")
    for i, cluster in enumerate(final_clusters):
        print(f"Cluster {i+1}: {len(cluster)} points")

    # Plot
    plot_clusters(final_clusters, final_centroids)
