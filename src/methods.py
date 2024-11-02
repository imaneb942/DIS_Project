import numpy as np
import scipy.sparse
import numba
import pandas as pd
import stopwordsiso
import Stemmer
import re
from tqdm import tqdm
import collections
import json
from src.utils import all_query_token_score
import pickle
import os
from typing import List, Dict, Tuple


class TFIDF:
    def __init__(
        self,
        corpus: List[str],
        save_path: str = None,
        load: bool = False,
        k1: float = None,
        b: float = None,
        d: float = None,
    ) -> None:
        """
        Implementation of TF-IDF scoring

        Attributes:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index (set during indexing)
            num_unique: int -> Number of unique tokens in the corpus

        """
        self.save_path = save_path
        self.corpus = corpus
        if load:
            self.load()
        else:
            self.num_corpus = len(self.corpus)

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/TFIDF"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)

        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)
        with open(f"{path}/params.json", "w") as f:
            json.dump({"num_corpus": self.num_corpus, "num_unique": self.num_unique}, f)

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/TFIDF"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the TF-IDF scores. We have the following implementation:
        TF[t, d] = count(t, d) / len(d)
        IDF[t] = log((1 + num_docs)/(1 + DF[t])) + 1
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """
        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        # DF[mickey] = #docs in which mickey comes
        # IDF[mickey] = log((1 + num_docs)/(1 + DF[mickey])) + 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # len(tfidf_scores) = sum(DF.values()) = total #tokens in all documents
        # for example, if doc1 has 3 tokens, doc2 has 4 tokens; then doc_indices is [1, 1, 1, 2, 2, 2, 2]

        """
        Terms x Documents grid:

            t1 t2 t3 t4 t5 t6 ... 
        d1
        d2
        d3

        Score[d1, t1] = TF[t1, d1] * IDF[t1]

        Seeing the above grid, we can use a sparse matrix to store the scores since most of the values are 0
        """

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)  # {word: count}
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )  # TF

            weighted_tf = (
                doc_token_counts / num_tokens
            )  # TF[token, document] / length(document)
            tfidf = (
                weighted_tf * self.IDF[doc_token_indices]
            )  # TF[token, document] * IDF[token]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((1 + num_docs)/(1 + DF[t])) + 1

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((1 + self.num_corpus) / (1 + DF[token])) + 1
        return IDF

    def parallel_search(
        self,
        query_tokens: List[List],
        query_token_map: Dict,
        topk: int = 10,
        num_threads: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the query tokens and the query_token_map, we calculate the topk documents for each query
        This is done in parallel using numba, and the number of threads is num_threads
        However, in kaggle, we don't get many cpu cores, but locally you can set num_threads to the number of cores

        Args:
            query_tokens: List[List] -> List of query tokens
            query_token_map: Dict -> Mapping of query token to index
            topk: int -> Number of documents to return for each query
            num_threads: int -> Number of threads to use for parallel computation
        """
        inverse_query_token_map = {v: k for k, v in query_token_map.items()}
        query_tokens = [
            [inverse_query_token_map[x] for x in query] for query in query_tokens
        ]
        query_token_ids = [
            [self.token_map[x] for x in query if x in self.token_map]
            for query in query_tokens
        ]
        # [[tokens of q1], [tokens of q2], ...]

        query_tokens_ids_flat = np.concatenate(query_token_ids).astype(np.int64)
        # [q1t1, q1t2, ...., q2t1, ...]

        query_ptrs = np.cumsum([0] + [len(x) for x in query_token_ids], dtype=np.int64)
        # [0, len(q1), len(q2) ...]

        numba.set_num_threads(num_threads)
        topk_scores, topk_indices = all_query_token_score(
            query_ptrs,
            query_tokens_ids_flat,
            topk,
            self.tfidf_indptr,
            self.tfidf_indices,
            self.tfidf_data,
            self.num_corpus,
        )
        numba.set_num_threads(1)
        return topk_indices, topk_scores


class BM25_V1(TFIDF):
    def __init__(
        self,
        corpus: List[str],
        k1: float = None,
        b: float = None,
        save_path: str = None,
        load: bool = False,
        d: float = None,
    ) -> None:
        """
        BM25 algorithm [1].

        In BM25_V1, we set:
            TF[t] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg)) where L_avg = sum(len(d) for d in corpus) / len(corpus)
            IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> BM25_V1 parameter
            b: float -> BM25_V1 parameter
            d: float -> Redundant parameter for BM25_V1

        References:
            [1] Robertson, Stephen & Walker, Steve & Jones, Susan & Hancock-Beaulieu, Micheline & Gatford, Mike. (1994). Okapi at TREC-3.. 0-.
        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/BM25_V1"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)
        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/BM25_V1"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)
        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the TF-IDF scores as per BM25_V1. We have the following implementation:
        TF[t, d] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg)) where L_avg = sum(len(d) for d in corpus) / len(corpus)
        IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )

            weighted_tf = doc_token_counts / (
                doc_token_counts + self.k1 * (1 - self.b * (1 - num_tokens / L_avg))
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: dict):
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log(
                (self.num_corpus - DF[token] + 0.5) / (DF[token] + 0.5) + 1
            )
        return IDF


class BM25_V2(BM25_V1):
    def __init__(
        self, corpus, k1=None, b=None, save_path=None, load=False, d=None
    ) -> None:
        """
        Modification of BM25_V1.
        In BM25_V2, we set the IDF[t] = log((1 + num_corpus)/(1 + DocFreq[t])) + 1

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> BM25_V2 parameter
            b: float -> BM25_V2 parameter
            d: float -> Redundant parameter for BM25_V2

        """
        super().__init__(corpus, k1, b, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/BM25_V2"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)
        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/BM25_V2"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((1 + num_corpus)/(1 + DF[t])) + 1

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((1 + self.num_corpus) / (1 + DF[token])) + 1
        return IDF


class BM25_PLUS(TFIDF):
    def __init__(
        self, corpus, k1=None, b=None, d=None, save_path=None, load=False
    ) -> None:
        """
        BM25_PLUS algorithm [2].
        In BM25_PLUS, we set:
            TF[t] = d + ((1 + k1) * count(t, d)) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
            IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> BM25_PLUS parameter
            b: float -> BM25_PLUS parameter
            d: float -> BM25_PLUS parameter

        References:
            [2] Lv, Yuanhua and ChengXiang Zhai. “Lower-bounding term frequency normalization.” International Conference on Information and Knowledge Management (2011).
        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b
            self.d = d

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/BM25_PLUS"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)

        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                    "d": self.d,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/BM25_PLUS"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]
            self.d = params["d"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the BM25_PLUS scores. We have the following implementation:
        TF[t, d] = d + ((1 + k1) * count(t, d)) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
        IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )

            # weighted_tf = doc_token_counts / (doc_token_counts + self.k1 * (1 - self.b*(1 - num_tokens/L_avg)))
            weighted_tf = self.d + (
                ((1 + self.k1) * doc_token_counts)
                / (doc_token_counts + self.k1 * (1 - self.b * (1 - num_tokens / L_avg)))
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((num_corpus + 1)/DF[t])

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((self.num_corpus + 1) / DF[token])
        return IDF


class BM25_PLUS_V2(TFIDF):
    def __init__(
        self,
        corpus: List[str],
        k1: float = None,
        b: float = None,
        d: float = None,
        save_path: str = None,
        load: bool = False,
    ) -> None:
        """
        Modification of the original BM25+ algorithm.
        In BM25_PLUS_V2, we set:
            TF[t] = d + ((1 + k1) * count(t, d)) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
            IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> BM25_PLUS_V2 parameter
            b: float -> BM25_PLUS_V2 parameter
            d: float -> BM25_PLUS_V2 parameter

        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b
            self.d = d

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/BM25_PLUS_V2"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)

        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                    "d": self.d,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/BM25_PLUS_V2"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)
        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]
            self.d = params["d"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the BM25_PLUS_V2 scores as per our modification. We have the following implementation:
        TF[t, d] = d + ((1 + k1) * count(t, d)) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
        IDF[t] = log((num_corpus + 1)/DF[t])
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )

            # weighted_tf = doc_token_counts / (doc_token_counts + self.k1 * (1 - self.b*(1 - num_tokens/L_avg)))
            weighted_tf = self.d + (
                ((1 + self.k1) * doc_token_counts)
                / (doc_token_counts + self.k1 * (1 - self.b * (1 - num_tokens / L_avg)))
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log(
                (self.num_corpus - DF[token] + 0.5) / (DF[token] + 0.5) + 1
            )
        return IDF


class DLITE(TFIDF):
    def __init__(
        self,
        corpus: List[str],
        k1: float = None,
        b: float = None,
        d: float = None,
        save_path: str = None,
        load: bool = False,
    ) -> None:
        """
        DLITE algorithm [3].
        In DLITE, we set:
            TF[t] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
            q_token =   DF[t] / num_docs
            q_prime_token = 1 - q_token
            IDF[t] = q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token)
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> DLITE parameter
            b: float -> DLITE parameter
            d: float -> Redundant parameter for DLITE

        References:
            [3] Ke, Weimao. "Alternatives to Classic BM25-IDF based on a New Information Theoretical Framework." 2022 IEEE International Conference on Big Data (Big Data). 2022.

        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b
            self.d = d

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/DLITE"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)

        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                    "d": self.d,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/DLITE"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]
            self.d = params["d"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the DLITE scores. We have the following implementation:
        TF[t, d] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
        q_token =   DF[t] / num_docs
        q_prime_token = 1 - q_token
        IDF[t] = q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token)
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index

        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )
            weighted_tf = doc_token_counts / (
                doc_token_counts + self.k1 * (1 - self.b * (1 - num_tokens / L_avg))
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        q_token =   DF[t] / num_docs
        q_prime_token = 1 - q_token
        IDF[t] = q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token)

        Args:
            DF: Dict -> Document Frequency of each token

        Returns:
            np.ndarray -> Inverse Document Frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            q_token = DF[token] / self.num_corpus
            q_prime_token = 1 - q_token
            IDF[token] = (
                q_prime_token / 2
                + (1 - q_token * (1 - np.log(q_token)))
                - (1 - (q_token**2) * (1 - 2 * np.log(q_token))) / (2 + 2 * q_token)
            )
        return IDF


class DLITE_CBRT(TFIDF):
    def __init__(
        self,
        corpus: List[str],
        k1: float = None,
        b: float = None,
        d: float = None,
        save_path: str = None,
        load: bool = False,
    ):
        """
        DLITE_cuberoot algorithm from DLITE [3].
        In DLITE_CBRT, we set:
            TF[t] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
            q_token =   DF[t] / num_docs
            q_prime_token = 1 - q_token
            IDF[t] = cbrt(q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token))
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> DLITE_CBRT parameter
            b: float -> DLITE_CBRT parameter
            d: float -> Redundant parameter for DLITE_CBRT
        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b
            self.d = d

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/DLITE_CBRT"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)

        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                    "d": self.d,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/DLITE_CBRT"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]
            self.d = params["d"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the DLITE_CBRT scores. We have the following implementation:
        TF[t, d] = count(t, d) / (count(t, d) + k1 * (1 - b + b * len(d) / L_avg))
        q_token =   DF[t] / num_docs
        q_prime_token = 1 - q_token
        IDF[t] = cbrt(q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token))
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )
            weighted_tf = doc_token_counts / (
                doc_token_counts + self.k1 * (1 - self.b * (1 - num_tokens / L_avg))
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        q_token =   DF[t] / num_docs
        q_prime_token = 1 - q_token
        IDF[t] = cbrt(q_prime_token/2 + (1 - q_token*(1-np.log(q_token))) - (1 - (q_token**2)*(1-2*np.log(q_token)))/(2 + 2*q_token))
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            q_token = DF[token] / self.num_corpus
            q_prime_token = 1 - q_token
            IDF[token] = (
                q_prime_token / 2
                + (1 - q_token * (1 - np.log(q_token)))
                - (1 - (q_token**2) * (1 - 2 * np.log(q_token))) / (2 + 2 * q_token)
            )
        return np.cbrt(IDF)


class TF_LDP(TFIDF):
    def __init__(
        self,
        corpus: List[str],
        k1: float = None,
        b: float = None,
        d: float = None,
        save_path: str = None,
        load: bool = False,
    ) -> None:
        """
        TF_L•D•P algorithm [4].

        In TF_LDP, we set:
            TF[t] = 1 + log(1 + log(d + (count(t, d) / (1 - b + b * len(d) / L_avg)))
            IDF[t] = log(num_corpus / DF[t])
            TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus: List[str] -> List of documents
            num_corpus: int -> Number of documents in the corpus
            save_path: str -> Path to save the TF-IDF scores
            token_map: Dict -> Mapping of token to index
            num_unique: int -> Number of unique tokens in the corpus
            k1: float -> TF_LDP parameter
            b: float -> TF_LDP parameter
            d: float -> TF_LDP parameter


        References:
            [4] Rousseau, François, Michalis, Vazirgiannis. "Composition of TF normalizations: new insights on scoring functions for ad hoc IR." Proceedings of the 36th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 2013.

        """
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b
            self.d = d

    def save(self) -> None:
        """
        Save the TF-IDF scores, class parameters, and corpus token mapping to the disk
        """
        path = f"{self.save_path}/TF_LDP"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/scores_data.npy", self.tfidf_data)
        np.save(f"{path}/scores_indices.npy", self.tfidf_indices)
        np.save(f"{path}/scores_indptr.npy", self.tfidf_indptr)
        with open(f"{path}/token_map.json", "w") as f:
            json.dump(self.token_map, f)

        with open(f"{path}/params.json", "w") as f:
            json.dump(
                {
                    "num_corpus": self.num_corpus,
                    "num_unique": self.num_unique,
                    "k1": self.k1,
                    "b": self.b,
                    "d": self.d,
                },
                f,
            )

    def load(self) -> None:
        """
        Load the TF-IDF scores, class parameters, and corpus token mapping from the disk
        """
        path = f"{self.save_path}/TF_LDP"
        self.tfidf_data = np.load(f"{path}/scores_data.npy")
        self.tfidf_indices = np.load(f"{path}/scores_indices.npy")
        self.tfidf_indptr = np.load(f"{path}/scores_indptr.npy")
        with open(f"{path}/token_map.json", "r") as f:
            self.token_map = json.load(f)

        with open(f"{path}/params.json", "r") as f:
            params = json.load(f)
            self.num_corpus = params["num_corpus"]
            self.num_unique = params["num_unique"]
            self.k1 = params["k1"]
            self.b = params["b"]
            self.d = params["d"]

    def calculate_scores(self, corpus_tokens: List[List], token_map: Dict) -> None:
        """
        Given the corpus tokens, and the token_map, we calculate the TF_LDP scores. We have the following implementation:

        TF[t, d] = 1 + log(1 + log(d + (count(t, d) / (1 - b + b * len(d) / L_avg)))
        IDF[t] = log(num_corpus / DF[t])
        TF-IDF[t, d] = TF[t, d] * IDF[t]

        Args:
            corpus_tokens: List[List] -> List of tokens for each document in the corpus
            token_map: Dict -> Mapping of token to index
        """

        self.token_map = token_map
        unique_tokens = list(token_map.values())
        self.num_unique = len(unique_tokens)
        L_avg = sum([len(x) for x in corpus_tokens]) / len(corpus_tokens)

        # Calculating Document Frequency
        set_unique_tokens = set(unique_tokens)
        DF = {x: 0 for x in set_unique_tokens}
        for document_tokens in corpus_tokens:
            tokens_present = set_unique_tokens.intersection(document_tokens)
            for token in tokens_present:
                DF[token] += 1

        self.IDF = self.calculate_idf(DF)

        tfidf_scores = np.empty(sum(DF.values()), dtype=np.float32)
        doc_indices = np.empty(sum(DF.values()), dtype=np.int64)
        word_indices = np.empty(sum(DF.values()), dtype=np.int64)

        # calculate tf-idf for each term
        ptr = 0
        for i, doc_tokens in enumerate(corpus_tokens):
            num_tokens = len(doc_tokens)
            doc_token_counts = collections.Counter(doc_tokens)
            doc_token_indices = np.array(list(doc_token_counts.keys()), dtype=np.int64)
            doc_token_counts = np.array(
                list(doc_token_counts.values()), dtype=np.float32
            )
            weighted_tf = 1 + np.log(
                1
                + np.log(
                    self.d
                    + (doc_token_counts / (1 - self.b + self.b * (num_tokens / L_avg)))
                )
            )
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr : ptr + len(doc_token_indices)] = tfidf
            doc_indices[ptr : ptr + len(doc_token_indices)] = i
            word_indices[ptr : ptr + len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix(
            (tfidf_scores, (doc_indices, word_indices)),
            shape=(self.num_corpus, self.num_unique),
            dtype=np.float32,
        )
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF: Dict) -> np.ndarray:
        """
        Calculate the inverse document frequency of each token
        IDF[t] = log((num_corpus + 1) / DF[t])
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((self.num_corpus + 1) / DF[token])

        return IDF
