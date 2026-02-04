import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering

def compute_clusters(
    dataset,
    n_clusters,
    active_features=None,
    top_n=5,
):
    """
    dataset: Dataset object
    n_clusters: number of clusters
    active_features: list of feature indices to filter down to before clustering
    top_n: number of top features to consider

    Returns:
        Dictionary mapping cluster IDs to cluster info:
        {
            cluster_id: {
                'top_examples': list of document indices,
                'rest_examples': list of remaining document indices,
                'top_scores': list of scores for top examples,
                'rest_scores': list of scores for remaining examples,
                'total_examples': total number of examples in cluster,
                'distinctive_features': {
                    'positive': [(feature_idx, label, score), ...],
                    'negative': [(feature_idx, label, score), ...]
                } or None
            }
        }
    """
    activations = dataset.latents() # (n_documents, n_features)
    feature_labels = dataset.feature_labels()

    filtered = activations[:, active_features] if active_features is not None else activations

    # --- Jaccard affinity ---
    bin_csr = csr_matrix(filtered > 0, dtype=np.int32)
    inter = bin_csr @ bin_csr.T
    row_sz = inter.diagonal()
    aff = np.array(np.nan_to_num(inter.toarray() / (row_sz[:, None] + row_sz - inter)))

    # --- Spectral clustering ---
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
    ).fit_predict(aff)

    # --- In-cluster affinity scores ---
    np.fill_diagonal(aff, 0)
    probs = np.zeros(len(aff))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size > 1:
            probs[idx] = aff[np.ix_(idx, idx)].sum(1) / (idx.size - 1)

    # --- Distinctive features ---
    def map_feat(i):
        return int(active_features[i]) if active_features is not None else int(i)

    distinctive = None
    if feature_labels is not None:
        distinctive = {}
        binary = (filtered > 0).astype(float)
        valid = labels != -1

        for c in sorted(set(labels[valid])):
            in_c = labels == c
            out_c = valid & ~in_c
            diffs = binary[in_c].mean(0) - binary[out_c].mean(0)

            order = np.argsort(diffs)
            top = order[-top_n:][::-1]
            bot = order[:top_n]

            def build(indices):
                out = []
                for i in indices:
                    orig = map_feat(i)
                    label = (
                        feature_labels[orig]
                        if orig in feature_labels
                        else f"feature_{orig}"
                    )
                    out.append((orig, label, float(diffs[i])))
                return out

            distinctive[c] = {
                "positive": build(top),
                "negative": build(bot),
            }

    # --- Cluster examples ---
    results = {}
    doc_ids = np.arange(len(activations))
    valid_clusters = sorted(set(labels) - {-1})

    for c in valid_clusters:
        mask = labels == c
        ids = doc_ids[mask]
        p = probs[mask]
        order = np.argsort(p)[::-1]

        results[c] = {
            "top_examples": list(ids[order][:top_n]),
            "rest_examples": list(ids[order][top_n:]),
            "top_scores": [float(x) for x in p[order][:top_n]],
            "rest_scores": [float(x) for x in p[order][top_n:]],
            "total_examples": len(ids),
            "distinctive_features": distinctive.get(c) if distinctive else None,
        }
    return results
