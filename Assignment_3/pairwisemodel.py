import torch.nn as nn
import ranking as rnk
import dataset
import evaluate as eval
import numpy as np
import torch
from tqdm import tqdm


class pairWiseModel(nn.Module):
    def __init__(self, num_features, scoring_network_layers, dropout=0.3, sigma=1, pairs=2000):
        super().__init__()
        self.scoringModules = nn.ModuleList()   # list with NN layers
        input_size = num_features
        for layer_size in scoring_network_layers:   # dynamic number of layers (for tuning)
            self.scoringModules.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(dropout), nn.ReLU()))    # append sequential: linear, dropout, relu
            input_size = layer_size     # change input_size to layer_size for input next hidden layer
        self.name = "Pairwise LTR model"
        self.scoringModules.append(nn.Linear(layer_size, 1))    # output layer
        self.scoringModules.to("cpu")   # store NN layers on CPU
        self.loss_fn = torch.nn.BCELoss()   # loss function binary cross entropy
        self.sigma = sigma
        self.pairs = pairs  # number of pairs

    def forward(self, docs):    # forward function
        scores = torch.Tensor(docs.feature_matrix)  # make tensor from feature matrix
        scores = scores.to("cpu")   # store scores on CPU
        for layer in self.scoringModules:
            scores = layer(scores)      # give scores to every layer in scoringModules
        self.scores = scores    # store scores in class
        return scores.detach().numpy().squeeze()    # convert tensor to numpy 1D array

    def rank(self, scores, docs):
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)    # get rankings and inverted rankings from scores
        return self.all_rankings, self.all_inverted_rankings

    def calc_ranknet(self, s_i, s_j):
        # computing cross entropy between desired probability and predicted probablity
        return 1 / (1 + torch.exp(-self.sigma * (s_i - s_j)))

    def loss_function(self, target):
        # initiliaze loss and target
        target = target.squeeze()
        loss = torch.zeros(1)

        # get pairs of docs for query
        for pair in range(self.pairs):
            # pick two docs at random
            i, j = np.random.choice(self.scores.shape[0], 2)
            # scores for document i and j
            s_i = self.scores[i]
            s_j = self.scores[j]
            # Sij is a element of {-1, 0, 1}, with -1 indicating that j is the more relevant document, 1 indicating i
            # is more relevant and 0 indicating equal relevancy.
            if target[i] > target[j]:
                S = 1
            elif target[i] == target[j]:
                S = 0
            else:
                S = -1

            S = torch.Tensor([S])
            P = self.calc_ranknet(s_i, s_j)
            loss += self.loss_fn(P, S)  # update loss

        return loss


def trainModel(model, data, epochs, optimizer):     # train model function
    print("======================== " + model.name + "========================")

    labels = torch.Tensor(data.train.label_vector).to("cpu").unsqueeze(1)   # save train labels
    labels_val = torch.Tensor(data.validation.label_vector).to("cpu").unsqueeze(1)  # save validation labels

    for epoch in tqdm(range(epochs)):
        model.train()   # turn on training mode
        optimizer.zero_grad()   # set gradients to zero
        model(data.train)   # train model with train data
        loss = model.loss_function(labels)      # get loss from loss function
        print('loss: ', loss)
        loss.backward()     # backpropagate loss
        optimizer.step()    # update weights
        if epoch % 10 == 0:
            print("validation ndcg at epoch " + str(epoch))
            model.eval()    # turn on evaluation mode
            validation_scores = model(data.validation)      # get validation scores from trained model
            results = eval.evaluate(data.validation, validation_scores, print_results=False)    # get results from evaluate function
            print(results["ndcg"])


# load dataset
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
# set number of epochs
epochs = 100
# initiate pairwise model
model = pairWiseModel(data.num_features, [10, 10, 10])
# adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# train model
trainModel(model, data, epochs, optimizer)

# get test scores from trained model
test_scores = model(data.test)
print('------')
print('Evaluation on entire test partition.')
# get results from test dataset and print results
results = eval.evaluate(data.test, test_scores, print_results=True)
