"""Semantic clustering for cross-domain belief selection.

Embeds belief texts using sentence-transformers and clusters with KMeans
to enable cross-cluster sampling in the derive pipeline.

Requires: pip install 'reasonsforge[cluster]'
"""

import random
from hashlib import sha256

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
    HAS_CLUSTER_DEPS = True
except ImportError:
    SentenceTransformer = None
    np = None
    KMeans = None
    HAS_CLUSTER_DEPS = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _require_cluster_deps():
    if not HAS_CLUSTER_DEPS:
        raise ImportError(
            "sentence-transformers and scikit-learn are required for --cluster mode. "
            "Install them with: pip install 'reasonsforge[cluster]'"
        )


class ClusterCache:
    """Cache embeddings across derive rounds for --exhaust mode."""

    def __init__(self, model_name=DEFAULT_MODEL):
        _require_cluster_deps()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._cache = {}

    def embed(self, beliefs):
        """Embed belief texts, using cache for previously seen beliefs.

        Args:
            beliefs: {node_id: text} dict

        Returns:
            (ids_list, embeddings_array) — ordered consistently
        """
        ids = sorted(beliefs.keys())
        uncached_ids = []
        uncached_texts = []

        for nid in ids:
            text = beliefs[nid]
            key = (nid, sha256(text.encode()).hexdigest()[:16])
            if key not in self._cache:
                uncached_ids.append(nid)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, show_progress_bar=False)
            for i, nid in enumerate(uncached_ids):
                text = beliefs[nid]
                key = (nid, sha256(text.encode()).hexdigest()[:16])
                self._cache[key] = new_embeddings[i]

        embeddings = []
        for nid in ids:
            text = beliefs[nid]
            key = (nid, sha256(text.encode()).hexdigest()[:16])
            embeddings.append(self._cache[key])

        return ids, np.array(embeddings)


def _auto_k(n_beliefs, n_clusters=None, max_k=20):
    """Compute cluster count from belief count or explicit override."""
    if n_clusters is not None:
        k = max(n_clusters, 1)
    else:
        k = n_beliefs // 5
        k = min(k, max_k)
        k = max(k, 2)
    return min(k, n_beliefs)


def cluster_beliefs(beliefs, budget, seed=None, n_clusters=None,
                    cache=None, model_name=DEFAULT_MODEL):
    """Cluster beliefs and sample across cluster boundaries.

    Args:
        beliefs: {node_id: text} for IN non-derived beliefs
        budget: max beliefs to select
        seed: random seed for KMeans and sampling
        n_clusters: override automatic cluster count
        cache: optional ClusterCache for embedding reuse
        model_name: sentence-transformers model name

    Returns:
        (selected_ids, cluster_stats) where cluster_stats contains
        n_clusters, cluster_sizes, and embedding_model.
    """
    _require_cluster_deps()

    if len(beliefs) <= budget:
        return list(beliefs.keys()), {
            "n_clusters": 1,
            "cluster_sizes": [len(beliefs)],
            "embedding_model": model_name,
        }

    if cache is None:
        cache = ClusterCache(model_name)

    ids, embeddings = cache.embed(beliefs)

    k = _auto_k(len(beliefs), n_clusters, max_k=min(budget // 3, 20))

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters = {}
    for i, nid in enumerate(ids):
        clusters.setdefault(labels[i], []).append(nid)

    cluster_sizes = [len(clusters[c]) for c in sorted(clusters)]

    rng = random.Random(seed)
    base_per = budget // k
    remainder = budget % k

    selected = []
    sorted_labels = sorted(clusters, key=lambda c: -len(clusters[c]))
    for i, label in enumerate(sorted_labels):
        members = clusters[label]
        alloc = base_per + (1 if i < remainder else 0)
        alloc = min(alloc, len(members))
        selected.extend(rng.sample(members, alloc))

    return selected, {
        "n_clusters": k,
        "cluster_sizes": cluster_sizes,
        "embedding_model": cache.model_name,
    }


def cluster_beliefs_intra(beliefs, budget, round_num=0, seed=None,
                          n_clusters=None, cache=None,
                          model_name=DEFAULT_MODEL):
    """Cluster beliefs and focus budget on one cluster per round.

    Rotates through clusters via round_num % k, giving the LLM
    topically adjacent beliefs that are more likely to combine.

    Returns:
        (selected_ids, cluster_stats) with focus_cluster index.
    """
    _require_cluster_deps()

    if len(beliefs) <= budget:
        return list(beliefs.keys()), {
            "n_clusters": 1,
            "cluster_sizes": [len(beliefs)],
            "embedding_model": model_name,
            "focus_cluster": 0,
        }

    if cache is None:
        cache = ClusterCache(model_name)

    ids, embeddings = cache.embed(beliefs)

    k = _auto_k(len(beliefs), n_clusters, max_k=min(budget // 3, 20))

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters = {}
    for i, nid in enumerate(ids):
        clusters.setdefault(labels[i], []).append(nid)

    cluster_sizes = [len(clusters[c]) for c in sorted(clusters)]

    sorted_labels = sorted(clusters, key=lambda c: -len(clusters[c]))
    focus_idx = round_num % k
    focus_label = sorted_labels[focus_idx]
    members = clusters[focus_label]

    rng = random.Random(seed)
    alloc = min(budget, len(members))
    selected = rng.sample(members, alloc)

    return selected, {
        "n_clusters": k,
        "cluster_sizes": cluster_sizes,
        "embedding_model": cache.model_name,
        "focus_cluster": focus_idx,
    }


def list_clusters(beliefs, n_clusters=None, seed=None, cache=None,
                  model_name=DEFAULT_MODEL):
    """Cluster beliefs and return full cluster assignments.

    Args:
        beliefs: {node_id: text} dict
        n_clusters: override automatic cluster count
        seed: random seed for KMeans
        cache: optional ClusterCache for embedding reuse
        model_name: sentence-transformers model name

    Returns:
        {"clusters": [{"id": int, "beliefs": [{"id": str, "text": str}]}],
         "n_clusters": int, "embedding_model": str}
    """
    _require_cluster_deps()

    if len(beliefs) <= 3:
        return {
            "clusters": [{"id": 0, "beliefs": [
                {"id": nid, "text": text} for nid, text in sorted(beliefs.items())
            ]}],
            "n_clusters": 1,
            "embedding_model": model_name,
        }

    if cache is None:
        cache = ClusterCache(model_name)

    ids, embeddings = cache.embed(beliefs)

    k = _auto_k(len(beliefs), n_clusters)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    groups = {}
    for i, nid in enumerate(ids):
        groups.setdefault(int(labels[i]), []).append(
            {"id": nid, "text": beliefs[nid]}
        )

    clusters = [
        {"id": label, "beliefs": members}
        for label, members in sorted(groups.items(), key=lambda x: -len(x[1]))
    ]

    return {
        "clusters": clusters,
        "n_clusters": k,
        "embedding_model": cache.model_name,
    }
