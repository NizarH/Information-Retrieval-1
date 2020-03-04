import numpy as np

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

def format_results(results_dict, type_run, fname, n_docs_limit=1000):
    # query-id Q0 document-id rank score STANDARD
    ranks = []
    for qid in results_dict:
        for i, doc_id in enumerate(results_dict[qid]):
            if i < n_docs_limit:
                rank = f"{qid} Q0 {doc_id} {i + 1} {results_dict[qid][doc_id]:.2} {type_run}"
                ranks.append(rank)
            else:
                break
    with open(f"./results/TREC_files/{fname}.txt", 'w') as f:
        for rank in ranks:
            f.write("%s\n" % rank)