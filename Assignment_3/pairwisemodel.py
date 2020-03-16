import torch.nn as nn
import ranking as rnk
import dataset
import evaluate as eval
import numpy as np
import torch
from tqdm import tqdm
device = 'cpu'

class pairWiseModel(nn.Module):
    def __init__(self, num_features, nn_layers, dropout=0.3, sigma=1, pairs=2000):
        super().__init__()
        self.scoring_n = nn.ModuleList() #get scoring layers for nn
        input_size = num_features
        for layer_size in nn_layers: # dynamic number of layers (for tuning)
            #forward pass in the nn
            self.scoring_n.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(dropout), nn.ReLU())) # append sequential: linear, dropout, relu
            input_size = layer_size # change input_size to layer_size for input next hidden layer
        self.name = "Pairwise LTR model"
        self.scoring_n.append(nn.Linear(layer_size, 1)) # output layer
        self.scoring_n.to(device) # store NN layers on CPU (device, can also be cuda if you have geforce)
        self.loss_fn = torch.nn.BCELoss() # loss function binary cross entropy for ranknet (later)
        self.sigma = sigma
        self.pairs = pairs # for pairwise LTR we need to specify the number of pairs we want to construct for ordering

    def score(self, docs): #forward function for scoring
        scores = torch.Tensor(docs.feature_matrix) # make tensor from feature matrix, which is a transformation on the dataset
        scores = scores.to(device) # store scores on CPU
        for layer in self.scoring_n:
            scores = layer(scores) # layer the scores for every every layer in all layers of our nn
        self.scores = scores # store scores in class
        return scores.detach().numpy().squeeze() # convert tensor to numpy 1D array

    def rank(self, scores, docs):
        # get rankings and inverted rankings from scores (from the predefined function)
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)
        return self.all_rankings, self.all_inverted_rankings


    def calc_ranknet(self, s_i, s_j):
        #computing cross entropy between desired probability and predicted probablity
        return 1 / (1 + torch.exp(-self.sigma * (s_i - s_j)))

    def loss_function(self, target):
        #initiliaze loss and target
        target = target.squeeze()
        loss = torch.zeros(1)

        ##get pairs of docs for query
        for pair in range(self.pairs):
            #pick two docs at random
            i, j = np.random.choice(self.scores.shape[0], 2)
            #scores for document i and j
            s_i = self.scores[i]
            s_j = self.scores[j]
            print(self.scores.shape[0])
            #Sij is a element of {-1, 0, 1}, with -1 indicating that j is the more relevant document, 1 indicating i
            # is more relevant and 0 indicating equal relevancy.
            if target[i] > target[j]:
                S = 1
            elif target[i] == target[j]:
                S = 0
            else:
                S = -1

            #save S in a tensor
            S = torch.Tensor([S])
            #calculate P using both document scores, with our earlier defined ranknet formula
            P = self.calc_ranknet(s_i, s_j)
            #get loss
            loss += self.loss_fn(P, S)

        return loss


def trainModel(model, data, epochs, optimizer):
    print("======================== " + model.name + "========================")

    labels = torch.Tensor(data.train.label_vector).to(device).unsqueeze(1)
    labels_val = torch.Tensor(data.validation.label_vector).to(device).unsqueeze(1)

    for epoch in tqdm(range(epochs)):
        model.train() # turn on training mode
        optimizer.zero_grad() # set gradients to zero
        model.score(data.train) # train model with train data
        loss = model.loss_function(labels) # get loss from loss function
        loss.backward() # backpropagate loss
        optimizer.step() # update weights
        if epoch%10 == 0:
            print("validation ndcg at epoch " + str(epoch))
            model.eval() # turn on evaluation mode
            validation_scores = model.score(data.validation)  # get validation scores from trained model
            results = eval.evaluate(data.validation, validation_scores, print_results=False) # evaluate model on validation data/scores
            print(results["ndcg"])



#load the dataset, fold it, and transform it for our model
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
#define how many times the model will 'see'the dataset for learning, in the form of number of epochs
epochs = 10
#pairwise model:
model = pairWiseModel(data.num_features, [10,10,10])
#choose an optimizer (we choose adam for best performance)
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

# get test scores from trained model
test_scores = model.score(data.test)
print('------')
print('Evaluation on entire test partition.')
# get results from test dataset and print results
results = eval.evaluate(data.test, test_scores, print_results=True)

