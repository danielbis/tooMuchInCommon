import logging

from sklearn.decomposition import PCA
import numpy as np

# create logger
module_logger = logging.getLogger('embeddings_benchmark.post_processing')


def all_but_the_top(embeddings: np.ndarray, top_d: int = 2, use_fix: bool = False) -> np.ndarray:
    """

    Source: https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390
    All-but-the-Top: Simple and Effective Postprocessing for Word Representations
    Paper: https://arxiv.org/abs/1702.01417
    Last Updated: Fri 15 Nov 2019 11:47:00 AM CET

    This function is an adaptation of the post-processing code from the link above.
    The variable names were changed, type annotation added.

    We extended the post-processing method to subtract mean centered embeddings
      projected onto the `top_d` principal components, instead of subtracting uncentered embeddings
      projected onto the `top_d` principal components (as was done in the original).
      Our version can be used with the argument `use_fix` = True.

    Args:
        embeddings: word vectors of shape (n_words, n_dimensions)
        top_d: number of principal components to subtract
        use_fix: when True subtracts mean centered embeddings projected onto the `top_d` principal components,
            else subtracts uncentered embeddings projected onto the `top_d` principal components
    Returns:
        np.ndarray: post-processed embeddings matrix of shape (n_words, n_dimensions)
    """
    # 1. Subtract mean vector
    # if scale:
    # module_logger.info("Removing mean scaled proportionally to the original vector's norm!")
    centered_embeddings = remove_mean(embeddings=embeddings, scale=False)
    # 2. Compute the first `top_d` principal components
    #    on centered embedding vectors
    u = PCA(n_components=top_d).fit(centered_embeddings).components_  # [D, emb_size]
    # Subtract first `top_d` principal components
    # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
    return centered_embeddings - (centered_embeddings @ u.T @ u) if use_fix \
        else centered_embeddings - (embeddings @ u.T @ u)


def remove_mean(embeddings: np.ndarray, scale: bool = False) -> np.ndarray:
    """

    Args:
        embeddings: word vectors of shape (n_words, n_dimensions)
        scale: if True, scaled mean direction will be subtracted --
                (1) Each vectors L2 norm will be computed
                (2) Vectors will be normalized to unit length before computing the mean direction,
                (3) Mean vector will be scaled by each vector's norm, from (1), before subtracting it
             else, mean vector of the `embeddings` matrix will be subtracted

    Returns:
        np.ndarray: 'mean centered' embeddings matrix of shape (n_words, n_dimensions)

    """
    if scale:
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit_embeddings = embeddings / embedding_norms
        mean_vec = embedding_norms * np.mean(unit_embeddings, axis=0, keepdims=True)
    else:
        mean_vec = np.mean(embeddings, axis=0)

    return embeddings - mean_vec


def identity(embeddings=None):
    return embeddings


