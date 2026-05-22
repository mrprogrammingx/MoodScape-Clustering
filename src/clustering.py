from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_best_k(X, min_k=2, max_k=8):
    best_k = 2
    best_score = -1

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def cluster_data(X, k):
    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = model.fit_predict(X)

    return model, labels
