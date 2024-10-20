import numpy as np
import scipy.sparse
import numba
import pandas as pd
import stopwordsiso
import Stemmer
import re
from tqdm import tqdm
import collections

class WordToBase:
    """
    Class to convert words to their base form
    """
    def __init__(self, lang):
        self.lang = lang

    def get_stemmer(self, single=False):
        try:
            if single:
                return Stemmer.Stemmer(self.lang).stemWord
            return Stemmer.Stemmer(self.lang).stemWords
        except Exception as e:
            # print(f'Inbuilt Stemmer does not exist for {self.lang}, creating one!')
            stemmer = self.create_stemmer(single)
        return stemmer
    
    # to be overwritten -> korean doesn't have stemmer from PyStemmer
    def create_stemmer(self, single=False):
        if single:
            return Stemmer.Stemmer('en').stemWord
        return Stemmer.Stemmer('en').stemWords


class TextProcessor:
    """
    Class to preprocess the corpus and queries
    """
    def __init__(self, lang, stopwords=None):
        self.lang = lang
        self.word_to_base = {}
        self.base_to_baseidx = {}
        self.word_to_wordidx = {}
        self.remove_punct = re.compile(r'(?u)\b\w\w+\b') # This is what sklearn uses
        if stopwords is None:
            self.stopwords = set(self.get_stopwords(lang))
        else:
            self.stopwords = stopwords

    def get_stopwords(self, lang):
        """
        Stopwords for the corresponding language
        """
        return stopwordsiso.stopwords(lang)
        

    def preprocess_corpus(self, corpus):
        """
        Given a corpus, which is a list of documents, we do the following for each document:
            - Convert document to lowercase
            - For all words in the document, we remove punctuations
            - If the word is a stopword, we discard it
            - Each word is converted to its base form using the stemmer
            - We create several mappings:
                • self.word_to_base: word -> base_word
                • self.base_to_baseidx: base_word -> base_idx (serves as the vocabulary)
                • self.word_to_wordidx: word -> base_idx
        We then return a list of lists, where each list is a document, and each element in the list is the base_idx of the word
        """
        stemmer = WordToBase(self.lang).get_stemmer(single=True)
        corpus_token_indices = []
        for doc in tqdm(corpus):
            document_token_indices = []
            doc = doc.lower()
            words = list(self.remove_punct.findall(doc))

            for word in words:
                # if we re-encounter the word, we don't need to recompute the base word
                if word in self.word_to_wordidx:
                    document_token_indices.append(self.word_to_wordidx[word])
                    continue
                # if we encounter a stopword, we discard it
                # Note, this if condition is 2nd as it improves performance
                if word in self.stopwords:
                    continue
                
                # if we have already computed the base word, we use it
                if word in self.word_to_base:
                    base_word = self.word_to_base[word]
                # otherwise, we compute the base word
                else:
                    base_word = stemmer(word)
                    self.word_to_base[word] = base_word
                # if we have already computed the base_idx, we use it
                if base_word in self.base_to_baseidx:
                    base_idx = self.base_to_baseidx[base_word]
                    self.word_to_wordidx[word] = base_idx
                    document_token_indices.append(base_idx)
                # else we compute the base_idx and update the mappings
                else:
                    base_idx = len(self.base_to_baseidx)
                    self.base_to_baseidx[base_word] = base_idx
                    self.word_to_wordidx[word] = base_idx
                    document_token_indices.append(base_idx)
            corpus_token_indices.append(document_token_indices)

        return corpus_token_indices, self.base_to_baseidx
    
    def preprocess_queries(self, queries):
        """ 
        Given a list of queries, we do the following for each query:
            - Convert query to lowercase
            - For all words in the query, we remove punctuations
            - If the word is a stopword, we discard it
            - We first form a collection of all the words in the queries
            - We then compute the base form of each word
            - Then we update all mappings, so that we can convert the query words to base_idx
        """
        query_token_ids = []
        word_to_idx = {}
        stemmer = WordToBase(self.lang).get_stemmer(single=False)
        for query in queries:
            query = query.lower() 
            words = self.remove_punct.findall(query)
            query_ids = []
            for word in words:
                # If we encounter a stopword, we discard it
                if word in self.stopwords:
                    continue
                # If we have not seen the word before, we update the mappings
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
                
                # Append the word_idx to the query_ids
                word_idx = word_to_idx[word]
                query_ids.append(word_idx)
            query_token_ids.append(query_ids)

        # After the above computation, we have a collection of all query words
        # Note that, it is not trivial to use the corpus vocabulary, since queries can have unseen words
        # We now compute the base form of each word and update the mappings

        # We first get the unique words in the queries
        unique_words = list(word_to_idx.keys())
        # We then compute the base form of each word
        base_words = stemmer(unique_words)
        unique_base_words = set(base_words)

        # We generate the base_word -> base_idx mapping
        unique_base_to_baseidx = {x:i for (i,x) in enumerate(unique_base_words)}

        # We create a mapping from word_idx -> base_idx
        wordidx_to_baseidx = {word_to_idx[word]:unique_base_to_baseidx[base] for (word, base) in zip(unique_words, base_words)}

        # Finally, we convert all to corresponding base words
        for i, query_tokens in enumerate(query_token_ids):
            query_token_ids[i] = [wordidx_to_baseidx[x] for x in query_tokens]

        return query_token_ids, unique_base_to_baseidx
