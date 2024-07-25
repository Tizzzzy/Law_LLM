from typing import List
from typing import List, Union, Dict, Any, Optional
import numpy as np
import os

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """
    Get the key for the prompt input.

    :param inputs: The input dictionary.
    :type inputs: Dict[str, Any]
    :param memory_variables: List of memory variables.
    :type memory_variables: List[str]
    :return: The key for the prompt input.
    :rtype: str
    :raises ValueError: If more than one input key is found.
    """
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected, got {prompt_input_keys}")
    return prompt_input_keys[0]


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """
    Get a value from a dictionary or an environment variable.

    :param key: The key to search in the dictionary.
    :type key: str
    :param env_key: The environment variable key.
    :type env_key: str
    :param default: Default value if the key is not found. Defaults to None.
    :type default: Optional[str], optional
    :return: The value associated with the key.
    :rtype: str
    :raises ValueError: If the key is not found and no default value is provided.
    """
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable "
            f"`{env_key}` which contains it, or pass "
            f"`{key}` as a named parameter."
        )


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """
    Get a value from a dictionary or an environment variable.

    :param data: The input dictionary.
    :type data: Dict[str, Any]
    :param key: The key to search in the dictionary.
    :type key: str
    :param env_key: The environment variable key.
    :type env_key: str
    :param default: Default value if the key is not found. Defaults to None.
    :type default: Optional[str], optional
    :return: The value associated with the key.
    :rtype: str
    """
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """
    Calculate row-wise cosine similarity between two equal-width matrices.

    :param X: The first matrix.
    :type X: Matrix
    :param Y: The second matrix.
    :type Y: Matrix
    :return: The cosine similarity matrix.
    :rtype: np.ndarray
    :raises ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. "
            f"X has shape {X.shape} and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """
    Calculate maximal marginal relevance.

    :param query_embedding: The query embedding.
    :type query_embedding: np.ndarray
    :param embedding_list: The list of embeddings.
    :type embedding_list: list
    :param lambda_mult: The lambda multiplier. Defaults to 0.5.
    :type lambda_mult: float, optional
    :param k: The number of embeddings to select. Defaults to 4.
    :type k: int, optional
    :return: The indices of the selected embeddings.
    :rtype: List[int]
    :raises ValueError: If the number of embeddings to select is invalid.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs