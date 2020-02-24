import os

import torch
import gensim
from pprint import pprint
import pickle as pkl
import read_ap
import download_ap
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ensure dataset is downloaded
download_ap.download_dataset()
# pre-process the text
docs_by_id = read_ap.get_processed_docs()


def divide_test_train(docs, doc_train="train_docs", doc_test="test_docs"):
    path_train = f"./pickles/{doc_train}.pkl"
    path_test = f"./pickles/{doc_test}.pkl"

    if not os.path.exists(path_train) or not os.path.exists(path_test):

        test_docs_len = len(list(docs.keys())) // 10
        train_docs, test_docs = {}, {}
        for i, doc_id in enumerate(docs):
            if i < test_docs_len:
                test_docs[doc_id] = docs[doc_id]
            else:
                train_docs[doc_id] = docs[doc_id]

        with open(path_train, "wb") as writer:
            pkl.dump(train_docs, writer)

        with open(path_test, "wb") as writer:
            pkl.dump(test_docs, writer)

        return train_docs, test_docs

    else:
        print("Docs already processed. Loading from disk")

        with open(path_train, 'rb') as reader_train, open(path_test, 'rb') as reader_test:
            return pkl.load(reader_train), pkl.load(reader_test)


def read_corpus(docs, tokens_only=False):
    for i, key in enumerate(docs):
        if tokens_only:
            yield docs[key]
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(docs[key], [i])


train_docs, test_docs = divide_test_train(docs_by_id)


train_corpus = list(read_corpus(train_docs))
test_corpus = list(read_corpus(test_docs, tokens_only=True))
pprint(train_corpus[:2])
print(test_corpus[:2])


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=50, epochs=10)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

vector = model.infer_vector(read_ap.process_text('president ronald reagan flew to kuwait for diplomatic meeting'))
print(vector)


# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#
#     second_ranks.append(sims[1])


# read in the qrels
qrels, queries = read_ap.read_qrels()
