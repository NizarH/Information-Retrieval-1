import read_ap
from pprint import pprint
from gensim import corpora
import gensim
import numpy as np
from gensim.matutils import kullback_leibler
import random

np.random.seed(1)

##BUILDING CORPUS, ADDITIONAL PREPROCESSING
NUM_TOPICS = 500
docs = read_ap.get_processed_docs()
len_docs = len(list(docs.keys())) // 5000
new_docs = dict(random.sample(docs.items(), len_docs))
print(len(new_docs))

d = w = 1
q = 0.5

leftovers = ["'s", "''", "``"]
#pprint(docs.keys())

texts = []
for k, v in new_docs.items():
    for word in v:
        if word in leftovers:
            v.remove(word)
    texts.append(v)

####LSI MODEL USING BOW FOR AQ3.1 specifically:

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# lsimodel_BoW = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)  # initialize an LSI transformation
# topics = lsimodel_BoW.print_topics(num_words=5)
# print('Most significant topics for LSI-BoW')
# for topic in topics:
#     print(topic)


## uncomment for most relevant topics for LSI: BoW


####LSI MODEL USING TF-IDF FOR AQ3.2 specifically:

tf_idf = gensim.models.TfidfModel(corpus) #BoW -> TF-IDF
corpus_tfidf = tf_idf[corpus] # transform the entire corpus
# lsimodel_TFIDF = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)  # initialize an LSI transformation
# topics = lsimodel_TFIDF.print_topics(num_words=5)
# print('Most significant topics for LSI-TFIDF')
# for topic in topics:
#     print(topic)

## uncomment for most relevant topics for LSI: TF-IDF


####LDA MODEL USING BOW FOR AQ3.3 specifically:

#lda model training using BoW. The Head-TA said LDA was fine using BoW as well (because TF-IDF doesn't work for 500 topics)
# ldamodel = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
# topics = ldamodel.print_topics(num_words=5)
# print("Most significant topics for LDA")
# for topic in topics:
#     print(topic)

### uncomment for most relevant topics for LDA: BoW



#AQ3.4 and generalizing AQ3.1 till AQ3.3:
#train LSI or LDA model:

def LSI_model(corpus, id2word=dictionary, num_topics=NUM_TOPICS):
    lsi = gensim.models.LsiModel(corpus, id2word=id2word, num_topics=num_topics)
    return lsi, "num_topics={}".format(num_topics)

def LDA_model(corpus, id2word=dictionary, num_topics=NUM_TOPICS):
    lda = gensim.models.LdaModel(corpus, id2word=id2word, num_topics=num_topics)
    return lda, "num_topics={}".format(num_topics)

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
            vec_lsi = model[vec_bow]  # convert the query to LSI space
            sims = index[vec_lsi] # get index
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            # store the scores with the associated doc id's for the retrieval evaluation
            doc_ids = list(new_docs.keys())
            for i, s in sims:
                sims_list.append((doc_ids[i], np.float64(s)))
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
            #same as with LSI BoW
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            doc_ids = list(new_docs.keys())
            for i, s in sims:
                sims_list.append((doc_ids[i], np.float64(s)))
            return sims_list
    else:
        #calculating the negative Kullback–Leibler divergence scores for LDA
        #transform query
        vec_bow = dictionary.doc2bow(processed_query)
        # transform query to the LDA space
        vec_lda_query = model[vec_bow][0]
        kl_divergence = []
        for text in corpus:
           #transform current document text in bow space to lda space
           vec_lda_text = model[text][0]
           # KL(Q||D) =\sum_w p(w|Q) log p(w|D) as explained in http://times.cs.uiuc.edu/course/410s11/kldir.pdf, using gensim mathutil
           kl_divergence.append(kullback_leibler(vec_lda_query, vec_lda_text))

        #sims = index[vec_lda]

        #sort the kl scores
        kl_divergence = sorted(enumerate(kl_divergence), key=lambda item: -item[1])
        doc_ids = list(new_docs.keys())
        for i, s in kl_divergence:
            sims_list.append((doc_ids[i], np.float64(s)))
        return sims_list

#tests for a single query
if __name__ == "__main__":
    # print("test")
    # # LSI BOW model
    # lsi_bow = LSI_model(corpus)[0]
    # test1 = rank_documents(lsi_bow, "LSI", "bow", "Design of the Star Wars Anti-missile Defense System")
    # print("OKE! Resultaten van LSI voor BoW: ")
    # pprint(test1[:10])
    #
    # lsi_tfidf = LSI_model(corpus_tfidf)[0]
    # test2 = rank_documents(lsi_tfidf, "LSI", "tfidf", "Design of the Star Wars Anti-missile Defense System")
    # print("OKE! Resultaten van LSI voor TFIDF: ")
    # pprint(test2[:10])

    lda_bow = LDA_model(corpus)[0]
    test3 = rank_documents(lda_bow, "LDA", "bow", "Design of the Star Wars Anti-missile Defense System")
    print("OKE! Resultaten van LDA voor BoW: ")
    pprint(test3[:10])