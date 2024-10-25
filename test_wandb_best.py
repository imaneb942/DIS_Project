import bm25s
import Stemmer
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from src.utils import idx_to_docid, return_recall_at_10, get_class
from src.methods import *
from src.text import TextProcessor
import wandb
import sys

if __name__ == '__main__':
    best_methods = {
        'ar': 'DLITE',
        'de': 'TF_LDP',
        'en': 'TF_LDP',
        'es': 'TF_LDP',
        'fr': 'BM25_V2',
        'it': 'BM25_V2',
        'ko': 'BM25_V1'
    }

    best_params = {
        'ar': {'k1': 1.6, 'b': 0.9, 'd': 1},
        'de': {'k1': 1.5, 'b': 0.9, 'd': 0.4},
        'en': {'k1': 1.5, 'b': 0.5, 'd': 0.8},
        'es': {'k1': 1.5, 'b': 0.8, 'd': 0.4},
        'fr': {'k1': 1.9, 'b': 0.8, 'd': 1},
        'it': {'k1': 1.4, 'b': 0.9, 'd': 1},
        'ko': {'k1': 1.5, 'b': 0.7, 'd': 1}
    }

 

    print('Loading Corpus')
    corpus = json.load(open('../dis-project-1-document-retrieval/corpus.json/corpus.json', 'r'))
    dev_df = pd.read_csv('../dis-project-1-document-retrieval/dev.csv')
    all_languages = set([x['lang'] for x in corpus])

    all_corpus = {}
    all_text = {}
    all_stemmers = {}
    all_tfidf = {}
    all_P = {}
    # for method_name in ['BM25_V1', 'BM25_V2', 'BM25_PLUS', 'BM25_PLUS_V2', 'DLITE', 'DLITE_CBRT']:
    # for method_name in ['TF_LDP']:
    method_name = 'TunedBest_Of_BM25_V1, BM25_V2, DLITE, TF_LDP'
    project_name = f'{method_name}_NormalStop'
    wandb.init(project='dis-project-1', name=project_name, entity='epfl-courses')

    
    
    print('Segragating and loading retrievers')
    for lang in all_languages:
        print('Starting', lang)
        my_corpus = [x for x in corpus if x['lang'] == lang]
        my_text = [x['text'] for x in my_corpus]
        all_corpus[lang] = my_corpus
        all_text[lang] = my_text
        # all_tfidf[lang] = BM25_V2(my_text, save_path=f'dump/{lang}', load=True)
        os.makedirs(f'best_dump/{lang}', exist_ok=True)
        method = get_class(f'src.methods.{best_methods[lang]}')
        all_tfidf[lang] = method(my_text, **best_params[lang], save_path=f'best_dump/{lang}')

        all_P[lang] = TextProcessor(lang)
        corpus_tokens, corpus_map = all_P[lang].preprocess_corpus(my_text)
        all_tfidf[lang].calculate_scores(corpus_tokens, corpus_map)
        all_tfidf[lang].save()
        

    all_queries = dev_df['query'].tolist()
    all_query_languages = dev_df['lang'].tolist()
    all_positive_docs = dev_df['positive_docs'].tolist()
    all_ids = dev_df.index.tolist()

    all_top_10_ids = []

    print(f'Running BM25')

    for query, lang in tqdm(zip(all_queries, all_query_languages)):
        query_tokens, query_map = all_P[lang].preprocess_queries([query])
        results, scores = all_tfidf[lang].parallel_search(query_tokens, query_map, topk=10, num_threads=1)
        results = idx_to_docid(results, all_corpus[lang])
        all_top_10_ids.append(results[0])

    overall_rec = return_recall_at_10(all_positive_docs, all_top_10_ids)
    wandb.log({'recall@10': overall_rec})
    # do for each language too
    for lang in all_languages:
        my_dev_df = dev_df[dev_df['lang'] == lang]
        my_dev_set_queries = my_dev_df['query'].tolist()
        my_dev_set_positive_docs = my_dev_df['positive_docs'].tolist()
        query_tokens, query_map = all_P[lang].preprocess_queries(my_dev_set_queries)
        results, scores = all_tfidf[lang].parallel_search(query_tokens, query_map, topk=10, num_threads=1)
        results = idx_to_docid(results, all_corpus[lang])
        s = return_recall_at_10(my_dev_set_positive_docs, results)

        wandb.log({f'{lang}_recall@10': s})

