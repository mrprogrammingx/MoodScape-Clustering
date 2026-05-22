import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(X, labels, output_path):
    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels
    )

    plt.title("Music Mood Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.colorbar(scatter)

    plt.savefig(output_path)
    plt.close()
