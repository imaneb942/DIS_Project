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

class TFIDF:
    def __init__(self, corpus, save_path=None, load=False):
        self.save_path = save_path
        if load:
            self.load()
        else:
            self.corpus = corpus 
            self.num_corpus = len(self.corpus)

    def save(self):
        """
        Save the TF-IDF scores to the disk
        """
        path = f'{self.save_path}/TFIDF'
        os.makedirs(path, exist_ok=True)
        np.save(f'{path}/scores_data.npy', self.tfidf_data)
        np.save(f'{path}/scores_indices.npy', self.tfidf_indices)
        np.save(f'{path}/scores_indptr.npy', self.tfidf_indptr)
        with open(f'{path}/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)
        with open(f'{path}/token_map.json', 'w') as f:
            json.dump(self.token_map, f)
        with open(f'{path}/params.json', 'w') as f:
            json.dump({'num_corpus': self.num_corpus, 'num_unique': self.num_unique}, f)
        
    def load(self):
        """ 
        Load the TF-IDF scores from the disk
        """
        path = f'{self.save_path}/TFIDF'
        self.tfidf_data = np.load(f'{path}/scores_data.npy')
        self.tfidf_indices = np.load(f'{path}/scores_indices.npy')
        self.tfidf_indptr = np.load(f'{path}/scores_indptr.npy')
        with open(f'{path}/token_map.json', 'r') as f:
            self.token_map = json.load(f)
        with open(f'{path}/corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)
        with open(f'{path}/params.json', 'r') as f:
            params = json.load(f)
            self.num_corpus = params['num_corpus']
            self.num_unique = params

    def calculate_scores(self, corpus_tokens, token_map):
        """
        Given the corpus tokens, and the token_map, we calculate the TF-IDF scores
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
            doc_token_counts = np.array(list(doc_token_counts.values()), dtype=np.float32)
            
            weighted_tf = doc_token_counts / num_tokens
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr:ptr+len(doc_token_indices)] = tfidf
            doc_indices[ptr:ptr+len(doc_token_indices)] = i
            word_indices[ptr:ptr+len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix((tfidf_scores, (doc_indices, word_indices)), shape=(self.num_corpus, self.num_unique), dtype=np.float32)
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF):
        """
        Calculate the inverse document frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((1 + self.num_corpus) / (1 + DF[token])) + 1
        return IDF
    
    def parallel_search(self, query_tokens, query_token_map, topk=10, num_threads=10):
        """
        Given the query tokens and the query_token_map, we calculate the topk documents for each query
        """
        inverse_query_token_map = {v: k for k, v in query_token_map.items()}
        query_tokens = [[inverse_query_token_map[x] for x in query] for query in query_tokens]
        query_token_ids = [[self.token_map[x] for x in query if x in self.token_map] for query in query_tokens]
        query_tokens_ids_flat = np.concatenate(query_token_ids).astype(np.int64)
        query_ptrs = np.cumsum([0] + [len(x) for x in query_token_ids], dtype=np.int64)

        numba.set_num_threads(num_threads)
        topk_scores, topk_indices = all_query_token_score(query_ptrs, query_tokens_ids_flat, topk, self.tfidf_indptr, self.tfidf_indices, self.tfidf_data, self.num_corpus)
        numba.set_num_threads(1)
        return topk_indices, topk_scores

class BM25_V1(TFIDF):
    """
    In BM25_V1, we set the IDF[t] = log((num_corpus - DF[t] + 0.5)/(DF[t] + 0.5) + 1)
    """
    def __init__(self, corpus, k1=None, b=None, save_path=None, load=False):
        super().__init__(corpus, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b

    def save(self):
        """
        Save the TF-IDF scores to the disk
        """
        path = f'{self.save_path}/BM25_V1'
        os.makedirs(path, exist_ok=True)
        np.save(f'{path}/scores_data.npy', self.tfidf_data)
        np.save(f'{path}/scores_indices.npy', self.tfidf_indices)
        np.save(f'{path}/scores_indptr.npy', self.tfidf_indptr)
        with open(f'{path}/token_map.json', 'w') as f:
            json.dump(self.token_map, f)
        with open(f'{path}/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)
        with open(f'{path}/params.json', 'w') as f:
            json.dump({'num_corpus': self.num_corpus, 'num_unique': self.num_unique, 'k1': self.k1, 'b': self.b}, f)
        
    def load(self):
        """ 
        Load the TF-IDF scores from the disk
        """
        path = f'{self.save_path}/BM25_V1'
        self.tfidf_data = np.load(f'{path}/scores_data.npy')
        self.tfidf_indices = np.load(f'{path}/scores_indices.npy')
        self.tfidf_indptr = np.load(f'{path}/scores_indptr.npy')
        with open(f'{path}/token_map.json', 'r') as f:
            self.token_map = json.load(f)
        with open(f'{path}/corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)
        with open(f'{path}/params.json', 'r') as f:
            params = json.load(f)
            self.num_corpus = params['num_corpus']
            self.num_unique = params['num_unique']
            self.k1 = params['k1']
            self.b = params['b']

    def calculate_scores(self, corpus_tokens, token_map):
        """
        Given the corpus tokens, and the token_map, we calculate the BM25 scores
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
            doc_token_counts = np.array(list(doc_token_counts.values()), dtype=np.float32)
            
            weighted_tf = doc_token_counts / (doc_token_counts + self.k1 * (1 - self.b*(1 - num_tokens/L_avg)))
            tfidf = weighted_tf * self.IDF[doc_token_indices]
            tfidf_scores[ptr:ptr+len(doc_token_indices)] = tfidf
            doc_indices[ptr:ptr+len(doc_token_indices)] = i
            word_indices[ptr:ptr+len(doc_token_indices)] = doc_token_indices
            ptr += len(doc_token_indices)

        tfidf_matrix = scipy.sparse.csc_matrix((tfidf_scores, (doc_indices, word_indices)), shape=(self.num_corpus, self.num_unique), dtype=np.float32)
        self.tfidf_data = tfidf_matrix.data
        self.tfidf_indices = tfidf_matrix.indices
        self.tfidf_indptr = tfidf_matrix.indptr

    def calculate_idf(self, DF):
        """
        Calculate the inverse document frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((self.num_corpus - DF[token] + 0.5)/(DF[token] + 0.5) + 1)
        return IDF
    
class BM25_V2(BM25_V1):
    """
    In BM25_V2, we set the IDF[t] = log((1 + num_corpus)/(1 + DocFreq[t])) + 1
    """
    def __init__(self, corpus, k1=None, b=None, save_path=None, load=False):
        super().__init__(corpus, k1, b, save_path, load)

        if not load:
            self.k1 = k1
            self.b = b

    def save(self):
        """
        Save the TF-IDF scores to the disk
        """
        path = f'{self.save_path}/BM25_V2'
        os.makedirs(path, exist_ok=True)
        np.save(f'{path}/scores_data.npy', self.tfidf_data)
        np.save(f'{path}/scores_indices.npy', self.tfidf_indices)
        np.save(f'{path}/scores_indptr.npy', self.tfidf_indptr)
        with open(f'{path}/token_map.json', 'w') as f:
            json.dump(self.token_map, f)
        with open(f'{path}/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)
        with open(f'{path}/params.json', 'w') as f:
            json.dump({'num_corpus': self.num_corpus, 'num_unique': self.num_unique, 'k1': self.k1, 'b': self.b}, f)
        
    def load(self):
        """ 
        Load the TF-IDF scores from the disk
        """
        path = f'{self.save_path}/BM25_V2'
        self.tfidf_data = np.load(f'{path}/scores_data.npy')
        self.tfidf_indices = np.load(f'{path}/scores_indices.npy')
        self.tfidf_indptr = np.load(f'{path}/scores_indptr.npy')
        with open(f'{path}/token_map.json', 'r') as f:
            self.token_map = json.load(f)
        with open(f'{path}/corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)
        with open(f'{path}/params.json', 'r') as f:
            params = json.load(f)
            self.num_corpus = params['num_corpus']
            self.num_unique = params['num_unique']
            self.k1 = params['k1']
            self.b = params['b']

    def calculate_idf(self, DF):
        """
        Calculate the inverse document frequency of each token
        """
        IDF = np.zeros(self.num_unique, dtype=np.float32)
        for token, _ in DF.items():
            IDF[token] = np.log((1 + self.num_corpus) / (1 + DF[token])) + 1
        return IDF
