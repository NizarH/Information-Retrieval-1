import read_ap
from pprint import pprint
import download_ap
from gensim import corpora
import gensim
import pickle

NUM_TOPICS = 500
docs = read_ap.get_processed_docs()
#leftovers = ["'s", "''", "``"]

texts = []
for k, v in docs.items():
    #for word in v:
    #    if word in leftovers:
      #      v.remove(word)
    texts.append(v)




####LSI MODEL USING BOW
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(word) for word in texts]

# lsimodel_bow = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
# topics = lsimodel_bow.print_topics(num_words=5)
# for topic in topics:
#     print(topic)


####LSI MODEL USING TF-IDF
tf_idf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tf_idf[corpus]

# lsi_model = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)  # initialize an LSI transformation
# lsimodel_TFIDF = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
# topics = lsi_model.print_topics(num_words=5)
# for topic in topics:
#     print(topic)


####LDA MODEL USING TF-IDF

#lda model training using tf_idf, note: ldamulticore is used to parallelize the process (faster training)
ldamodel = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)

def rank_documents(model, type, query):

    dictionary2 = corpora.Dictionary(texts)
    corpus2 = [dictionary2.doc2bow(text) for text in texts]
    tf_idf2 = gensim.models.TfidfModel(corpus2)
    corpus_tfidf2 = tf_idf2[corpus2]

    if model == "LSI":
        if type == "bow":
            lsi = gensim.models.LsiModel(corpus2, id2word=dictionary2, num_topics=500)
            index = gensim.similarities.MatrixSimilarity(lsi[corpus2])
            vec_bow = dictionary.doc2bow(query.lower().split())
            vec_lsi = lsi[vec_bow]  # convert the query to LSI space
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            for i, s in enumerate(sims):
                print(s, enumerate(docs)[i])
            return sims

        else:
            lsi_tfidf = gensim.models.LsiModel(corpus_tfidf2, id2word=dictionary, num_topics=NUM_TOPICS)
            index = gensim.similarities.MatrixSimilarity(lsi_tfidf[corpus_tfidf2])
            vec_bow = dictionary.doc2bow(query.lower().split())
            vec_lsi = lsi_tfidf[vec_bow]
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            for i, s in enumerate(sims):
                print(s, enumerate(docs)[i])
            return sims
    else:
        lda = gensim.models.LdaModel(corpus_tfidf2, id2word=dictionary, num_topics=NUM_TOPICS)
        index = gensim.similarities.MatrixSimilarity(lda[corpus_tfidf2])
        vec_bow = dictionary.doc2bow(query.lower().split())
        vec_lda = lda[vec_bow]
        sims = index[vec_lda]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for i, s in enumerate(sims):
            print(s, enumerate(docs)[i])
        return sims

