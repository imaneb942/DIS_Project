from src.methods import *
import json
import pandas as pd
import time
import numpy as np
from src.text import TextProcessor
import os
from src.utils import idx_to_docid, return_recall_at_10, get_class
import argparse

class ExplicitDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)

parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument(
    "--method",
    type=str,
    default="TFIDF",
    help="args.method to use for retrieval",
    choices=[
        "TFIDF",
        "BM25_V1",
        "BM25_V2",
        "BM25_PLUS",
        "BM25_PLUS_V2",
        "DLITE",
        "DLITE_CBRT",
        "TF_LDP",
    ],
)
parser.add_argument(
    "--save-index", action="store_true", help="Save the index and vocab"
)
parser.add_argument("--k1", type=float, default=1.5, help="k1 parameter")
parser.add_argument("--b", type=float, default=0.75, help="b parameter")
parser.add_argument("--d", type=float, default=1, help="delta parameter")
args = parser.parse_args()


if __name__ == "__main__":
    corpus = json.load(
        open("dis-project-1-document-retrieval/corpus.json/corpus.json", "r")
    )
    dev_df = pd.read_csv("dis-project-1-document-retrieval/dev.csv")


for lang in ["ar", "de", "es", "it", "ko", "en", "fr"]:
    print("Starting", lang)

    my_corpus = [x for x in corpus if x["lang"] == lang]
    my_text = [x["text"] for x in my_corpus]
    P = TextProcessor(lang)

    corpus_tokens, corpus_map = P.preprocess_corpus(my_text)
    my_dev_df = dev_df[dev_df["lang"] == lang]
    my_dev_set_queries = my_dev_df["query"].tolist()
    my_dev_set_positive_docs = my_dev_df["positive_docs"].tolist()
    query_tokens, query_map = P.preprocess_queries(my_dev_set_queries)
    os.makedirs(f"dump/{lang}", exist_ok=True)
    print("Starting", args.method)
    M = get_class(f"src.methods.{args.method}")

    tf_idf = M(
        my_text, k1=args.k1, b=args.b, d=args.d, save_path=f"dump/{lang}"
    )  # for TF_LDP
    print("Starting indexing")
    start = time.perf_counter()
    tf_idf.calculate_scores(corpus_tokens, corpus_map)
    end = time.perf_counter()
    print(f"Indexing took {end - start}s.")

    if args.save_index:
        print(f"Saving Index and Vocab to dump/{lang}")
        start = time.perf_counter()
        tf_idf.save()
        end = time.perf_counter()
        print(f"Saving index took {end - start}s.")
    results, scores = tf_idf.parallel_search(
        query_tokens, query_map, topk=10, num_threads=1
    )
    results = idx_to_docid(results, my_corpus)
    print(return_recall_at_10(my_dev_set_positive_docs, results))
