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
    print('Loading Corpus')
    corpus = json.load(open('../dis-project-1-document-retrieval/corpus.json/corpus.json', 'r'))
    dev_df = pd.read_csv('../dis-project-1-document-retrieval/dev.csv')
    all_languages = set([x['lang'] for x in corpus])

    all_corpus = {}
    all_text = {}
    all_stemmers = {}
    all_tfidf = {}
    all_P = {}
    for method_name in ['BM25_V1', 'BM25_V2', 'BM25_PLUS', 'BM25_PLUS_V2', 'DLITE', 'DLITE_CBRT', 'TF_LDP']:
        project_name = f'{method_name}_NormalStop'
        run = wandb.init(project='dis-project-1', name=project_name, entity='epfl-courses', reinit=True)

        method = get_class(f'src.methods.{method_name}')
        
        
        print('Segragating and loading retrievers')
        for lang in all_languages:
            my_corpus = [x for x in corpus if x['lang'] == lang]
            my_text = [x['text'] for x in my_corpus]
            all_corpus[lang] = my_corpus
            all_text[lang] = my_text
            # all_tfidf[lang] = BM25_V2(my_text, save_path=f'dump/{lang}', load=True)
            all_tfidf[lang] = method(my_text, save_path=f'dump/{lang}', load=True)
            all_P[lang] = TextProcessor(lang)


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
        run.finish()

