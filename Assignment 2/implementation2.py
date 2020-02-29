import read_ap
from pprint import pprint
import download_ap
from gensim import corpora
import gensim
import pickle
import numpy as np

NUM_TOPICS = 500
docs = read_ap.get_processed_docs()
len_docs = len(list(docs.keys())) // 1000
new_docs = {}
for i, doc_id in enumerate(docs):
    if i == len_docs:
        break
    new_docs[doc_id] = docs[doc_id]
#leftovers = ["'s", "''", "``"]
# pprint(docs.keys())

texts = []
for k, v in new_docs.items():
    #for word in v:
    #    if word in leftovers:
      #      v.remove(word)
    texts.append(v)


####LSI MODEL USING BOW
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(word) for word in texts]
# lsimodel_BoW = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)  # initialize an LSI transformation
# topics = lsimodel_BoW_TFIDF.print_topics(num_words=5)
# for topic in topics:
#     print(topic)

### uncomment for most relevant topics for LSI: BoW


####LSI MODEL USING TF-IDF
tf_idf = gensim.models.TfidfModel(corpus) #BoW -> TF-IDF
corpus_tfidf = tf_idf[corpus] # transform the entire corpus
# lsimodel_TFIDF = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)  # initialize an LSI transformation
# topics = lsimodel_TFIDF.print_topics(num_words=5)
# for topic in topics:
#     print(topic)

### uncomment for most relevant topics for LSI: TF-IDF


# ####LDA MODEL USING BOW
#
# #lda model training using BoW. The Head-TA said LDA was fine using BoW as well (because TF-IDF doesn't work for 500 topics)
# ldamodel = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
# topics = ldamodel.print_topics(num_words=5)
# print("LDA MODEL MET 500 TOPICS met BOW corpus")
# print("######################")
# for topic in topics:
#     print(topic)

### uncomment for most relevant topics for LDA: BoW

def LSI_model(corpus, id2word=dictionary, num_topics=NUM_TOPICS):
    lsi = gensim.models.LsiModel(corpus, id2word=id2word, num_topics=num_topics)
    return lsi, "num_topics={}".format(NUM_TOPICS)

def rank_documents(model, model_name, type, query):

    sims_list = []

    processed_query = read_ap.process_text(query)
    print(processed_query)

    if model_name == "LSI":
        if type == "bow":
            # calculating cosine similarity for LSI (BoW)
            index = gensim.similarities.MatrixSimilarity(model[corpus])
            #make a bow representation of the query, and split the words
            vec_bow = dictionary.doc2bow(processed_query)
            print(query.lower().split())
            vec_lsi = model[vec_bow]  # convert the query to LSI space
            sims = index[vec_lsi]
            # print(sims)
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            # store the scores with the associated doc id's for the retrieval evaluation
            for i, s in sims:
                doc_id = list(new_docs.keys())[i]
                sims_list.append((doc_id, np.float64(s)))
            return sims_list

        if type == "tfidf":
            #calculating cosine similarity for LSI, tf idf using similarities
            #use the tfidf corpus -> lsi corpus
            corpus_lsi = model[corpus_tfidf]
            #transform corpus to LSI space and index it
            index = gensim.similarities.MatrixSimilarity(corpus_lsi)
            #convert query to lsi space via tf-idf
            vec_bow = dictionary.doc2bow(processed_query)
            vec_lsi = model[vec_bow]
            sims = index[vec_lsi]
            # pprint(sims)
            #same as with LSI BoW
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            for i, s in sims:
                doc_id = list(new_docs.keys())[i]
                sims_list.append((doc_id, np.float64(s)))
            return sims_list
    else:
        #calculating the negative Kullback–Leibler divergence scores for LDA
        lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
        index = gensim.similarities.MatrixSimilarity(lda[corpus])
        vec_bow = dictionary.doc2bow(query.lower().split())
        vec_lda = lda[vec_bow]
        sims_index = index[vec_lda]
        sims = [(doc, gensim.matutils.kullback_leibler(doc, vec_lda)) for doc in sims_index]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for i, s in sims:
            doc_id = list(new_docs.keys())[i]
            sims_list.append((doc_id, np.float64(s)))
        return sims_list

if __name__ == "__main__":
    # LSI BOW model
    lsi_bow = LSI_model(corpus)[0]
    test1 = rank_documents(lsi_bow, "LSI", "bow", "Design of the Star Wars Anti-missile Defense System")
    print("OKE! Resultaten van LSI voor BoW: ")
    pprint(test1[:10])

    lsi_tfidf = LSI_model(corpus_tfidf)[0]
    test2 = rank_documents(lsi_tfidf, "LSI", "tfidf", "Design of the Star Wars Anti-missile Defense System")
    print("OKE! Resultaten van LSI voor TFIDF: ")
    pprint(test2[:10])
    # test3 = rank_documents("LDA", "BOW", "Design of the Star Wars Anti-missile Defense System")
    # print("OKE! Resultaten van LDA voor BoW: ")
    # pprint(test3[:10])

