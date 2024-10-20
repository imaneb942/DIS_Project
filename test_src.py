import bm25s
import Stemmer
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from src.utils import idx_to_docid, recall_at_10
from src.methods import *
from src.text import TextProcessor

if __name__ == '__main__':
    print('Loading Corpus')
    corpus = json.load(open('../dis-project-1-document-retrieval/corpus.json/corpus.json', 'r'))
    dev_df = pd.read_csv('../dis-project-1-document-retrieval/test.csv')
    all_languages = set([x['lang'] for x in corpus])

    all_corpus = {}
    all_text = {}
    all_stemmers = {}
    all_tfidf = {}
    all_P = {}
    print('Segragating and loading retrievers')
    for lang in all_languages:
        my_corpus = [x for x in corpus if x['lang'] == lang]
        my_text = [x['text'] for x in my_corpus]
        all_corpus[lang] = my_corpus
        all_text[lang] = my_text
        all_tfidf[lang] = BM25_V2(my_text, save_path=f'dump/{lang}', load=True)
        all_P[lang] = TextProcessor(lang)


    all_queries = dev_df['query'].tolist()
    all_query_languages = dev_df['lang'].tolist()
    # all_positive_docs = dev_df['positive_docs'].tolist()
    all_ids = dev_df.index.tolist()

    all_top_10_ids = []

    print(f'Running BM25')

    for query, lang in tqdm(zip(all_queries, all_query_languages)):
        query_tokens, query_map = all_P[lang].preprocess_queries([query])
        results, scores = all_tfidf[lang].parallel_search(query_tokens, query_map, topk=10, num_threads=1)
        results = idx_to_docid(results, all_corpus[lang])
        all_top_10_ids.append(results[0])



    # recall_at_10(all_positive_docs, all_top_10_ids)

    my_df = pd.DataFrame({'id': all_ids, 'docids': [str(x) for x in all_top_10_ids]})
    my_df.to_csv('submission.csv')


        









