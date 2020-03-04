import os
import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
import pytrec_eval
from tqdm import tqdm
from pprint import pprint
import random

import read_ap
import download_ap


def print_results(docs_by_id, results, query, n_docs_limit=10, len_limit=50):
    query_text = f"Query: {query}"
    print(query_text)
    results = results[:n_docs_limit]
    ranks = []
    ranks.append(query_text)
    for i, (doc_id, score) in enumerate(results):
        d = " ".join(docs_by_id[doc_id])
        doc_content = d[:len_limit] + "..."
        rank = f"\tRank {i}({score:.2}, {doc_id}): {doc_content}"
        ranks.append(rank)
        print(rank)
    return ranks


class TfIdfRetrieval:

    def __init__(self, docs):

        index_path = f"./pickles/tfidf_index_subsample"
        if os.path.exists(index_path):

            with open(index_path, "rb") as reader:
                index = pkl.load(reader)

            self.ii = index["ii"]
            self.df = index["df"]
        else:
            self.ii = defaultdict(list)
            self.df = defaultdict(int)

            doc_ids = list(docs.keys())

            print("Building Index")
            # build an inverted index
            for doc_id in tqdm(doc_ids):
                doc = docs[doc_id]

                counts = Counter(doc)
                for t, c in counts.items():
                    self.ii[t].append((doc_id, c))
                # count df only once - use the keys
                for t in counts:
                    self.df[t] += 1

            with open(index_path, "wb") as writer:
                index = {
                    "ii": self.ii,
                    "df": self.df
                }
                pkl.dump(index, writer)

    def search(self, query):
        query_repr = read_ap.process_text(query)

        results = defaultdict(float)
        for query_term in query_repr:
            if query_term not in self.ii:
                continue
            for (doc_id, tf) in self.ii[query_term]:
                results[doc_id] += np.log(1 + tf) / self.df[query_term]

        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results


if __name__ == "__main__":

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    len_docs = len(list(docs_by_id.keys())) // 5
    new_docs = dict(random.sample(docs_by_id.items(), len_docs))

    # Create instance for retrieval
    tfidf_search = TfIdfRetrieval(new_docs)
    # read in the qrels
    qrels, queries = read_ap.read_qrels()

    # try a random query
    random_qid = random.choice(list(queries.keys()))
    random_query = queries[random_qid]
    # get tf-idf scores for random query
    results = tfidf_search.search(random_query)
    # print results of random query
    print_results(new_docs, results, random_query)

    overall_ser = {}

    print("Running TFIDF Benchmark")
    # collect results
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results = tfidf_search.search(query_text)
        overall_ser[qid] = dict(results)

    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(f"./results/tf-idf_subsample.json", "w") as writer:
        json.dump(metrics, writer, indent=1)



