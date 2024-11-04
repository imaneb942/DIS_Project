from src.methods import *
import json
import pandas as pd
import time
import numpy as np
from src.text import TextProcessor
import sys
from src.utils import idx_to_docid, get_class, return_recall_at_10
import wandb


if __name__ == "__main__":
    corpus = json.load(
        open("dis-project-1-document-retrieval/corpus.json/corpus.json", "r")
    )
    dev_df = pd.read_csv("dis-project-1-document-retrieval/dev.csv")

    all_corpus = {}
    all_text = {}
    all_stemmers = {}
    all_tfidf = {}
    all_P = {}
    all_corpus_tokens = {}
    all_corpus_map = {}
    all_query_tokens = {}
    all_query_map = {}
    all_dev_set_positive_docs = {}
    all_dev_set_queries = {}
    for lang in ["ar", "de", "es", "it", "ko", "en", "fr"]:
        print("Starting", lang)
        my_corpus = [x for x in corpus if x["lang"] == lang]
        my_text = [x["text"] for x in my_corpus]
        P = TextProcessor(lang)
        all_corpus[lang] = my_corpus
        all_text[lang] = my_text
        corpus_tokens, corpus_map = P.preprocess_corpus(my_text)
        all_P[lang] = P
        my_dev_df = dev_df[dev_df["lang"] == lang]
        my_dev_set_queries = my_dev_df["query"].tolist()
        all_dev_set_queries[lang] = my_dev_set_queries
        my_dev_set_positive_docs = my_dev_df["positive_docs"].tolist()
        all_dev_set_positive_docs[lang] = my_dev_set_positive_docs
        query_tokens, query_map = P.preprocess_queries(my_dev_set_queries)
        all_corpus_tokens[lang] = corpus_tokens
        all_corpus_map[lang] = corpus_map
        all_query_tokens[lang] = query_tokens
        all_query_map[lang] = query_map

    bs = np.linspace(0.3, 0.9, 13)
    ks = np.linspace(1.0, 2.0, 11)
    ds = [1, 0.5]

    method = sys.argv[1]
    M = get_class(f"src.methods.{method}")
    print("Starting", method)
    for d in ds:
        for b in bs:
            for k1 in ks:
                print(f"Starting indexing for b={b}, d={d}, k1={k1}")
                project_name = f"{method}_b={b}_k1={k1}_d={d}_NormalStop"
                run = wandb.init(
                    project=method,
                    name=project_name,
                    entity="epfl-courses",
                    reinit=True,
                )
                wandb.config = {"b": b, "k1": k1, "d": d}
                for lang in ["ar", "de", "es", "it", "ko", "en", "fr"]:
                    tf_idf = M(
                        all_text[lang], k1=k1, b=b, d=d, save_path=f"dump/{lang}"
                    )
                    start = time.perf_counter()
                    tf_idf.calculate_scores(
                        all_corpus_tokens[lang], all_corpus_map[lang]
                    )
                    end = time.perf_counter()
                    print(f"Indexing took {end - start}s.")
                    results, scores = tf_idf.parallel_search(
                        all_query_tokens[lang],
                        all_query_map[lang],
                        topk=10,
                        num_threads=1,
                    )
                    results = idx_to_docid(results, all_corpus[lang])
                    s = return_recall_at_10(all_dev_set_positive_docs[lang], results)
                    print("Recall for lang", lang, "is", s)
                    wandb.log({f"{lang}_recall@10": s})
                run.finish()
