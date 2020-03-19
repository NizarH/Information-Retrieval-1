import torch.nn as nn
import ranking as rnk
import dataset
import evaluate as eval
import numpy as np
import torch
from tqdm import tqdm
from evaluate import ndcg_at_k
device = 'cpu'

#implement regular ERR for the ordinal relevancy scales for DELTA ERR (in loss function)
def ERR(labels):
    ERR = 0
    p = 1

    maximum_score = 4 #relevancy is incidated on a ordinal scale from 0 to 4, with 4 the highest (perfect label)
    ERR_max_val = maximum_score**2   #ERR maximum value for classificaton, second part of the formula
    theta = [(2 ** label_score - 1) / ERR_max_val for label_score in labels] #first part of the formula
    for r, R in enumerate(labels):
        #algorithm 2 in the paper
        ERR += p * theta[r] / (r+1)
        p *= (1-theta[r])
    return ERR



class listWiseModel(nn.Module):
    def __init__(self, num_features, nn_layers, dropout=0.3, sigma=1, pairs=2000, metric ="nDCG"):
        super().__init__()
        self.scoring_n = nn.ModuleList()
        input_size = num_features
        for layer_size in nn_layers:
            self.scoring_n.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(dropout), nn.ReLU()))
            input_size = layer_size
        self.name = "Listwise LTR model with " + metric
        self.scoring_n.append(nn.Linear(layer_size, 1))
        self.scoring_n.to(device)
        #self.loss_fn = torch.nn.BCELoss()
        self.sigma = sigma
        self.pairs = pairs
        #ERR or nDCG, standard = nDCG
        self.metric = metric

    def score(self, docs):
        scores = torch.Tensor(docs.feature_matrix)
        scores = scores.to(device)
        for layer in self.scoring_n:
            scores = layer(scores)
        self.scores = scores
        return scores.detach().numpy().squeeze()

    def rank(self, scores, docs):
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)
        return self.all_rankings, self.all_inverted_rankings


    def calc_ranknet(self, s_i, s_j):
        #computing cross entropy between desired probability and predicted probablity
        return 1 / (1 + torch.exp(-self.sigma * (s_i - s_j)))

    def loss_function(self, target):
        #initiliaze loss and target
        target = target.squeeze()
        loss = torch.zeros(1)

        # implementation of nDCG for delta nDCG, (based on ndcg_at_k with in evaluate (with no cutoff) but made for delta nDCG)
        max_range = lambda labels: len(labels) + 1
        DCG = lambda labels: sum([(1. / (np.log2(i + 2.))) * (2 ** labels[i-1]) for i in range(1, max_range(labels))])
        nDCG = lambda labels: DCG(labels) / ideal

        #Depending on given metric choose nDCG or ERR for delta irm,ij
        if self.metric == "nDCG":
            delta_irm_ij = nDCG
        if self.metric == "ERR":
            delta_irm_ij = ERR

        #as with pairwise decide on the number of the docs (instead of pairs) to generate from all docs
        doc_selection = np.random.choice(self.scores.shape[0], self.pairs)
        doc_labels = [float(target[i]) for i in doc_selection] #order the labels as a product of the chosen docs
        ideal_labels = np.sort(doc_labels)[::-1] #high to low, i.e the ideal order, needed for the for the nDCG formula
        ideal = DCG(ideal_labels)


        #implement Lambda(i,j)
        lambdarank = 0
        delta_metric = 0

        ##get pairs of docs for query, for swapping
        for doc_i, i in enumerate(doc_selection):
            for doc_j, j in enumerate(doc_selection):
                if i == j: continue  # do not go through the loop if the same document is selected

                # swap the scores
                swapped = doc_labels.copy()
                swapped[doc_i], swapped[doc_j] = swapped[doc_j], swapped[doc_i]

                #calculate |delta for chosen metric|, either ERR or NDCG
                delta_ij = delta_irm_ij(swapped) - delta_irm_ij(doc_labels)

                # get scores like with pairwise
                s_i = self.scores[i]
                s_j = self.scores[j]

                if target[i] > target[j]:
                    S = 1
                elif target[i] == target[j]:
                    S = 0
                else:
                    S = -1

                S = torch.Tensor([S])
                # detaching for lambdarank
                s_i_new = s_i.detach()
                s_j_new = s_j.detach()

                lambda_ij = float(self.sigma * (0.5 * (1 - S) - self.calc_ranknet(s_i_new, s_j_new))[0])
                # print(lambda_ij)

                delta_metric += np.abs(delta_ij)

                lambdarank += lambda_ij

            loss += lambdarank * delta_metric * s_i
            print(loss)
            return loss


def trainModel(model, data, epochs, optimizer):
    print("======================== " + model.name + "========================")

    labels = torch.Tensor(data.train.label_vector).to(device).unsqueeze(1)
    torch.Tensor(data.validation.label_vector).to(device).unsqueeze(1)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        model.score(data.train)
        loss = model.loss_function(labels)
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print("validation ndcg at epoch " + str(epoch))
            model.eval()
            validation_scores = model.score(data.validation)
            results = eval.evaluate(data.validation, validation_scores, print_results=False)
            print(results["ndcg"])




data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
epochs = 100

#nDCG
model = listWiseModel(data.num_features, [10,10,10], metric = "ERR")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# initialize a random model
trainModel(model, data, epochs, optimizer)

test_scores = model.score(data.test)
print('------')
print('Evaluation on entire test partition.')
results = eval.evaluate(data.test, test_scores, print_results=True)


##ERR
model = listWiseModel(data.num_features, [10,10,10], metric = "nDCG")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# initialize a random model
trainModel(model, data, epochs, optimizer)

test_scores = model.score(data.test)
print('------')
print('Evaluation on entire test partition.')
results = eval.evaluate(data.test, test_scores, print_results=True)

