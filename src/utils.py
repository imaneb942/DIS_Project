import numpy as np
import scipy.sparse
import numba
import pandas as pd
import stopwordsiso
import Stemmer
import re
from tqdm import tqdm
import collections
from typing import Sequence, List, Dict, Tuple


@numba.njit()
def sift_down(values, indices, startpos, pos):
    new_value = values[pos]
    new_index = indices[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent_value = values[parentpos]
        if new_value < parent_value:
            values[pos] = parent_value
            indices[pos] = indices[parentpos]
            pos = parentpos
            continue
        break
    values[pos] = new_value
    indices[pos] = new_index


@numba.njit()
def sift_up(values, indices, pos, length):
    startpos = pos
    new_value = values[pos]
    new_index = indices[pos]
    childpos = 2 * pos + 1
    while childpos < length:
        rightpos = childpos + 1
        if rightpos < length and values[rightpos] < values[childpos]:
            childpos = rightpos
        values[pos] = values[childpos]
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    values[pos] = new_value
    indices[pos] = new_index
    sift_down(values, indices, startpos, pos)


@numba.njit()
def heap_push(values, indices, value, index, length):
    values[length] = value
    indices[length] = index
    sift_down(values, indices, 0, length)


@numba.njit()
def heap_pop(values, indices, length):
    return_value = values[0]
    return_index = indices[0]
    last_value = values[length - 1]
    last_index = indices[length - 1]
    values[0] = last_value
    indices[0] = last_index
    sift_up(values, indices, 0, length - 1)
    return return_value, return_index


@numba.njit()
def parallel_topk(array: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the topk elements in the array in parallel using a heap.
    Implementation from https://github.com/xhluca/bm25s/blob/main/bm25s/numba/selection.py

    Args:
        array: np.ndarray -> Array of scores
        topk: int -> Number of topk elements to return

    Returns:
        Tuple[np.ndarray, np.ndarray] -> Tuple of topk scores and their indices
    """
    n = len(array)
    if topk > n:
        topk = n

    values = np.zeros(topk, dtype=array.dtype)  # aka scores
    indices = np.zeros(topk, dtype=np.int64)
    length = 0

    for i, value in enumerate(array):
        if length < topk:
            heap_push(values, indices, value, i, length)
            length += 1
        else:
            if value > values[0]:
                values[0] = value
                indices[0] = i
                sift_up(values, indices, 0, length)

    sorted_indices = np.flip(np.argsort(values))
    indices = indices[sorted_indices]
    values = values[sorted_indices]

    return values, indices


@numba.njit()
def query_token_score(single_query_tokens, score_indptr, indices, data, num_corpus):
    start = score_indptr[single_query_tokens]
    end = score_indptr[single_query_tokens + 1]
    scores = np.zeros(num_corpus, dtype=np.float32)
    for j in range(len(single_query_tokens)):
        _s, _e = start[j], end[j]
        for k in range(_s, _e):
            scores[indices[k]] += data[k]

    return scores


@numba.njit(parallel=True)
def all_query_token_score(
    query_ptrs, query_tokens_ids_flat, topk, score_indptr, indices, data, num_corpus
):
    topk_scores = np.zeros(
        (len(query_ptrs) - 1, topk), dtype=np.float32
    )  # num queries x k
    topk_indices = np.zeros((len(query_ptrs) - 1, topk), dtype=np.int64)

    for i in numba.prange(len(query_ptrs) - 1):
        single_query_tokens = query_tokens_ids_flat[query_ptrs[i] : query_ptrs[i + 1]]
        single_query_score = query_token_score(
            single_query_tokens, score_indptr, indices, data, num_corpus
        )
        topk_scores_sing, topk_indices_sing = parallel_topk(
            single_query_score, topk=topk
        )
        topk_scores[i] = topk_scores_sing
        topk_indices[i] = topk_indices_sing

    return topk_scores, topk_indices


def return_recall_at_10(positive_docs: List[str], top_10_ids: List[str]) -> float:
    """
    Given the positive_docs and top_10_ids, return the recall@10. Since we have only 1 positive document, this is equivalent to the check if the positive document is in the top 10 documents

    Args:
        positive_docs: List[str] -> List of docids of the positive documents
        top_10_ids: List[str] -> List of docids of top 10 documents for each query

    """
    recall = []
    for positive_doc, top_10_id in zip(positive_docs, top_10_ids):
        recall.append(positive_doc in top_10_id)
    return np.mean(recall)


def idx_to_docid(idx_list_list: List[List], corpus: List[Dict]) -> List[List[str]]:
    """
    Given idx_list_list, which is a list of indices of the topk documents corresponding to each query in the corpus, return the docids of the topk documents

    Args:
        idx_list_list: List[List] -> List of indices of the topk documents for each query
        corpus: List[Dict] -> List of documents in the corpus

    Returns:
        List[List[str]] -> List of docids of the topk documents for each query
    """
    res = []
    for idx_list in idx_list_list:
        res.append(list(map(lambda x: corpus[x]["docid"], idx_list)))
    return res


def get_class(kls: str) -> type:
    """
    Given a string representation of a class, return the class itself (eg. 'src.methods.BM25_V2' -> BM25_V2)

    Args:
        kls: str -> String representation of the class

    Returns:
        type -> Class itself
    """
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
