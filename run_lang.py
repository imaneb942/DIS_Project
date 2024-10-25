from src.methods import *
import json
import pandas as pd
import time
import numpy as np
from src.text import TextProcessor
import os
from src.utils import idx_to_docid, recall_at_10, get_class



if __name__ == '__main__':
    corpus = json.load(open('../dis-project-1-document-retrieval/corpus.json/corpus.json', 'r'))
    dev_df = pd.read_csv('../dis-project-1-document-retrieval/dev.csv')




for lang in ['ar', 'de', 'es', 'it', 'ko', 'en', 'fr']:
    # print('Starting', lang)
    my_corpus = [x for x in corpus if x['lang'] == lang]
    my_text = [x['text'] for x in my_corpus]

    P = TextProcessor(lang, stopwords_version='v2')
    corpus_tokens, corpus_map = P.preprocess_corpus(my_text)
    my_dev_df = dev_df[dev_df['lang'] == lang]
    my_dev_set_queries = my_dev_df['query'].tolist()
    my_dev_set_positive_docs = my_dev_df['positive_docs'].tolist()
    query_tokens, query_map = P.preprocess_queries(my_dev_set_queries)
    os.makedirs(f'dump/{lang}', exist_ok=True)
    # for method in ['BM25_V1', 'BM25_V2', 'BM25_PLUS', 'BM25_PLUS_V2', 'DLITE', 'DLITE_CBRT', 'TF_LDP']:
    for method in ['BM25_V2']:
        # print('Starting', method)
        M = get_class(f'src.methods.{method}')
        # tf_idf = M(my_text, k1=1.5, b=0.75, d=1, save_path=f'dump/{lang}') # for all 
        
        tf_idf = M(my_text, k1=1.5, b=0.75, d=1, save_path=f'dump/{lang}') # for TF_LDP
        # print('Starting indexing')
        start = time.perf_counter()
        tf_idf.calculate_scores(corpus_tokens, corpus_map)
        end = time.perf_counter()
        # print(f'Indexing took {end - start}s.')
        # print(f'Saving Index and Vocab to dump/{lang}')
        # start = time.perf_counter()
        # tf_idf.save()
        # end = time.perf_counter()
        # print(f'Saving index took {end - start}s.')
        results, scores = tf_idf.parallel_search(query_tokens, query_map, topk=10, num_threads=1)
        results = idx_to_docid(results, my_corpus)
        print(lang)
        recall_at_10(my_dev_set_positive_docs, results)