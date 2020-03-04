import os
import random

import numpy as np
import json
import pickle as pkl
import logging

import gensim
import pytrec_eval
from tqdm import tqdm

import read_ap
import download_ap
from helpers import format_results
from tf_idf import print_results
import lsi_lda

np.random.seed(1)


def subsample_train(docs, doc_train="train_docs"):
    path_train = f"./pickles/{doc_train}.pkl"

    if not os.path.exists(path_train):
        train_docs_len = len(list(docs.keys())) - len(list(docs.keys())) // 3
        train_docs = dict(random.sample(docs.items(), train_docs_len))
        with open(path_train, "wb") as writer:
            pkl.dump(train_docs, writer)
        return train_docs
    else:
        print("Docs already processed. Loading from disk")
        with open(path_train, 'rb') as reader_train:
            return pkl.load(reader_train)

def read_corpus(docs, keys_tags_dict, tokens_only=False):
    length_dict = len(list(keys_tags_dict.keys()))
    for i, key in enumerate(docs):
        if tokens_only:
            keys_tags_dict[length_dict + i] = key
            yield docs[key]
        else:
            keys_tags_dict[i] = key
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(docs[key], [i])


def build_and_train_doc2vec(train_corpus, vector_size=400, window=5, min_count=50, max_vocab_size=100000, workers=4, start_alpha=0.025,
                            end_alpha=0.005,
                            epochs=5):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, max_vocab_size=max_vocab_size, workers=workers, start_alpha=start_alpha,
                                          end_alpha=end_alpha, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # save model in pkl
    model_file_path = "./pickles/doc2vec_model_vecsize={}_window={}_maxvocab={}_epochs={}.pkl".format(vector_size, window, max_vocab_size, epochs)
    with open(model_file_path, "wb") as writer:
        pkl.dump(model, writer)

    return model, "vecsize={}_window={}_maxvocab={}_epochs={}".format(vector_size, window, max_vocab_size, epochs)


def get_average_score(scores):
    MAP_scores = []
    NDCG_scores = []
    map_key = 'map'
    ndcg_key = 'ndcg'

    for query in scores:
        MAP_scores.append(scores[query][map_key])
        NDCG_scores.append(scores[query][ndcg_key])

    avg_scores = {map_key: np.mean(MAP_scores), ndcg_key: np.mean(NDCG_scores)}
    return avg_scores


def run_benchmark(model, qrels, queries, IR_method, params, validation_set):
    path = f"./results/{IR_method[0]}_{params}_default.json"
    # if not os.path.exists(path):
    print("Running {} Benchmark".format(IR_method[0]))
    overall_ser = {}
    if IR_method[0] == "doc2vec":
        # collect results
        for i, qid in enumerate(tqdm(qrels)):
            if qid in validation_set:
                query_text = queries[qid]
                sims = IR_method[1](model, query_text)
                overall_ser[qid] = dict(sims)
    elif IR_method[0] == "LSI_BOW":
        # collect results
        for i, qid in enumerate(tqdm(qrels)):
            if qid not in validation_set:
                query_text = queries[qid]
                sims = IR_method[1](model, "LSI", "bow", query_text)
                overall_ser[qid] = dict(sims)
    elif IR_method[0] == "LSI_TFIDF":
        # collect results
        for i, qid in enumerate(tqdm(qrels)):
            if qid not in validation_set:
                query_text = queries[qid]
                sims = IR_method[1](model, "LSI", "tfidf", query_text)
                overall_ser[qid] = dict(sims)
    else:
        # collect results
        for i, qid in enumerate(tqdm(qrels)):
            # if qid not in validation_set:
            query_text = queries[qid]
            sims = IR_method[1](model, "LDA", "bow", query_text)
            overall_ser[qid] = dict(sims)

    format_results(overall_ser, f"{IR_method[0]}_tuned", f"{IR_method[0]}_{params}")

    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # get average score from MAP and NDCG
    avg_scores = get_average_score(metrics)
    print(avg_scores)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(path, "w") as writer:
        json.dump(metrics, writer, indent=1)

    return avg_scores


def doc2vec_search(model, query):
    processed_query = read_ap.process_text(query)
    inferred_vector = model.infer_vector(processed_query)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    sims = [(keys_tags_dict[doc_id], np.float64(score)) for (doc_id, score) in sims]
    return sims


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()
    # get qrels and queries
    qrels, queries = read_ap.read_qrels()
    # define list of IR search functions
    list_of_search_fns = [("doc2vec", doc2vec_search), ("LSI_BOW", lsi_lda.rank_documents), ("LSI_TFIDF", lsi_lda.rank_documents), ("LDA", lsi_lda.rank_documents)]

    # divide docs into train and test corpus
    train_docs = subsample_train(docs_by_id)
    # # save doc_ids and corresponding tags in dict
    keys_tags_dict = {}
    # # convert docs into TaggedDocuments
    train_corpus = list(read_corpus(train_docs, keys_tags_dict))

    # tune params
    vector_dims = [200, 300, 400, 500]
    window_sizes = [5, 10, 15, 20]
    vocab_sizes = [10000, 25000, 50000, 100000, 200000]
    validation_set = np.arange(76, 101)
    validation_set = list(map(str, validation_set))

    scores_per_param = {}
    for vocab_size in vocab_sizes:
        # build model and train it
        model, params = build_and_train_doc2vec(train_corpus)
        # free up some ram
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        # run benchmark
        avg_scores = run_benchmark(model, qrels, queries, list_of_search_fns[0], params, validation_set)
        scores_per_param[f"doc2vec_avg"] = avg_scores

    with open("./results/scores_validation_set.json", "w") as writer:
        json.dump(scores_per_param, writer, indent=1)
    num_topics = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    for num_topic in num_topics:
        lsi_bow, params = lsi_lda.LSI_model(lsi_lda.corpus)
        run_benchmark(lsi_bow, qrels, queries, list_of_search_fns[1], params, validation_set)
        scores_per_param[f"lsi-bow numtop={num_topic}"] = avg_scores

    with open("./results/lsi_bow_avgscores_tuned.json", "w") as writer:
        json.dump(scores_per_param, writer, indent=1)

    lsi_tfidf, params = lsi_lda.LSI_model(lsi_lda.corpus_tfidf)

    run_benchmark(lsi_tfidf, qrels, queries, list_of_search_fns[2], params, validation_set)

    with open("./results/lsi_tfidf_avgscores_default.json", "w") as writer:
        json.dump(avg_scores, writer, indent=1)

    lda_bow, params = lsi_lda.LDA_model(lsi_lda.corpus)

    avg_scores = run_benchmark(lda_bow, qrels, queries, list_of_search_fns[3], params, validation_set)

    with open("./results/lsa_bow_avgscores_default.json", "w") as writer:
        json.dump(avg_scores, writer, indent=1)


    # try random query
    qid = random.choice(list(queries.keys()))
    random_query = queries[qid]
    print(random_query)
    # get doc2vec scores for random query
    model_file_path = "./pickles/doc2vec_model_vecsize=400_window=5_maxvocab=100000_epochs=5.pkl"
    model = pkl.load(open(model_file_path, 'rb'))
    avg_scores = run_benchmark(model, qrels, queries, list_of_search_fns[0], "vecsize=400_window=5_maxvocab=100000_epochs=5", validation_set)
    with open("./results/doc2vec_avg_scores_76-100_tuned.json", "w") as writer:
        json.dump(avg_scores, writer, indent=1)
    results = doc2vec_search(model, random_query)
    # print results of random query
    ranks = print_results(docs_by_id, results, random_query)

