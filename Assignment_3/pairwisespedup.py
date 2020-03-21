import torch.nn as nn
import ranking as rnk
import dataset
import evaluate as eval
import numpy as np
import torch
from tqdm import tqdm
from pytorchtools import EarlyStopping
import pointwise as pnt
import pairwise as pr
import matplotlib.pyplot as plt

device = 'cpu'


class pairWiseModel(nn.Module):
    def __init__(self, num_features, nn_layers, dropout=0.1, sigma=0.1, pairs=2000):
        super().__init__()
        self.scoring_n = nn.ModuleList()  # get scoring layers for nn
        input_size = num_features
        for layer_size in nn_layers:  # dynamic number of layers (for tuning)
            # forward pass in the nn
            self.scoring_n.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(dropout),
                                                nn.ReLU()))  # append sequential: linear, dropout, relu
            input_size = layer_size  # change input_size to layer_size for input next hidden layer
        self.name = "Pairwise Sped Up LTR model"
        self.scoring_n.append(nn.Linear(layer_size, 1))  # output layer
        self.scoring_n.to(device)  # store NN layers on CPU (device, can also be cuda if you have geforce)
        # self.loss_fn = torch.nn.MSELoss() # loss function binary cross entropy for ranknet (later)
        self.sigma = sigma
        self.pairs = pairs  # for pairwise LTR we need to specify the number of pairs we want to construct for ordering

    def forward(self, docs):  # forward function for scoring
        scores = torch.Tensor(
            docs.feature_matrix)  # make tensor from feature matrix, which is a transformation on the dataset
        scores = scores.to(device)  # store scores on CPU
        for layer in self.scoring_n:
            scores = layer(scores)  # layer the scores for every every layer in all layers of our nn
        self.scores = scores  # store scores in class
        return scores.detach().numpy().squeeze()  # convert tensor to numpy 1D array

    def rank(self, scores, docs):
        # get rankings and inverted rankings from scores (from the predefined function)
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)
        return self.all_rankings, self.all_inverted_rankings

    def calc_ranknet(self, s_i, s_j):
        # computing cross entropy between desired probability and predicted probablity
        return 1 / (1 + torch.exp(-self.sigma * (s_i - s_j)))

    def sped_up_loss(self, target):
        # initiliaze loss and target
        target = target.squeeze()
        loss = torch.zeros(1)

        # initialize lambda_i
        lambda_i = 0

        # instead of with regular ranknet we iterate for two sets of documents
        # to get the pairs number we use np.sqrt(pairs) * np.sqrt(pairs) which should give us an equal number of iterations to the number of pairs defined
        docs_i = np.random.choice(self.scores.shape[0], int(np.ceil(np.sqrt(self.pairs))))
        docs_j = []

        pair_number = 0
        while pair_number != len(docs_i):
            random_doc = np.random.choice(self.scores.shape[0])
            if random_doc not in docs_i:
                docs_j.append(random_doc)
                pair_number += 1

        for i in docs_i:
            # score for document i
            s_i = self.scores[i]
            for j in docs_j:
                # scores for document j
                s_j = self.scores[j]
                # Sij is a element of {-1, 0, 1}, with -1 indicating that j is the more relevant document, 1 indicating i
                # is more relevant and 0 indicating equal relevancy.
                if target[i] > target[j]:
                    Sij = 1
                elif target[i] == target[j]:
                    Sij = 0
                else:
                    Sij = -1

                Sij = torch.Tensor([Sij])
                # as shown in equation 6
                lambda_ij = float(self.sigma * (0.5 * (1 - Sij) - self.calc_ranknet(s_i, s_j)))
                # as shown in equation 7
                lambda_i += lambda_ij

            # sum ∂si/ ∂wk * (sumj∈Pi * ∂C(si, sj) / ∂si) (= lambda_i) (equation 4/5)
            # weights update happens in train model
            loss += lambda_i * s_i

        return loss


def trainModel(model, data, epochs, optimizer, model_type, patience=10):
    print("======================== " + model.name + "========================")

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(model_type, patience=patience, verbose=True, delta=0.0001)

    labels = torch.Tensor(data.train.label_vector).to(device).unsqueeze(1)
    labels_val = torch.Tensor(data.validation.label_vector).to(device).unsqueeze(1)
    scores, train_losses, validation_losses, ndcg_list, average_relevant_rank = [], [], [], [], []

    for epoch in tqdm(range(epochs)):
        model.train()  # turn on training mode
        optimizer.zero_grad()  # set gradients to zero
        model(data.train)  # train model with train data
        loss = model.sped_up_loss(labels)
        loss.backward()  # backpropagate loss
        optimizer.step()  # update weights

        # if epoch % 10 == 0:
        train_losses.append(loss.item())
        print("validation ndcg at epoch " + str(epoch))
        model.eval()  # turn on evaluation mode
        validation_scores = model(data.validation)  # get validation scores from trained model
        scores.append(validation_scores)
        # valid_loss = model.sped_up_loss(labels_val)
        # validation_losses.append(valid_loss)
        results = eval.evaluate(data.validation, validation_scores,
                                print_results=False)  # evaluate model on validation data/scores
        ndcg_list.append(results['ndcg']['mean'])
        average_relevant_rank.append(results['relevant rank']['mean'])
        print('ndcg: ', results["ndcg"])
        print('relevant rank: ', results['relevant rank'])

            # epoch_len = len(str(epochs))
            # print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
            #              f'train_loss: {loss.item():.5f} ' +
            #              f'valid_loss: {valid_loss.item():.5f}')
            # print(print_msg)
            #
            # # early_stopping checks if validation loss has decresed
            # early_stopping(valid_loss.item(), model)
            #
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        # gc.collect()
    print('mean ndcg: ', np.mean(ndcg_list))
    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('models/{}_checkpoint.pt'.format(model_type)))

    return model, optimizer, scores, train_losses, validation_losses, ndcg_list, average_relevant_rank


if __name__ == "__main__":
    # load the dataset, fold it, and transform it for our model
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    # define how many times the model will 'see'the dataset for learning, in the form of number of epochs
    epochs = 80
    LR = 0.01
    num_nodes_layers = [10] * 25
    model_type = 'pairwise-spedup'
    # pairwise model:
    model = pairWiseModel(data.num_features, num_nodes_layers)
    # choose an optimizer (we choose adam for best performance)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    optim_str = optimizer.__str__()[0:4]

    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' % data.validation.num_queries())
    print('Number of documents in validation set: %d' % data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    # initialize a random model
    model, optimizer, validation_scores, train_loss, valid_loss, ndcg_list, average_relevant_rank = trainModel(model, data, epochs, optimizer, model_type)
    # pnt.plot_loss(valid_loss, 'ReLu', optim_str, LR, model_type, 'Loss dCt/dWk', epochs)
    pnt.plot_ndcg(ndcg_list, 'ReLu', optim_str, LR, model_type, epochs)
    # pnt.plot_earlystop(train_loss, valid_loss, model_type)
    pr.plot_ARR(average_relevant_rank, 'ReLU', optim_str, LR, model_type, epochs)

    # pnt.save_model(model, epochs, optimizer, validation_scores, train_loss, valid_loss, ndcg_list, average_relevant_rank, LR, 'ReLu', optim_str,
    #                model_type)
    # file_load = 'models/pairwise_spedup_ltr_tuned_eps=80_LR=0.01_ReLu_Adam_1.pth.tar'
    # model, optimizer, checkpoint, epochs, validation_scores, train_loss, valid_loss, ndcg_list, average_relevant_rank = pnt.load_model(
    #     model, optimizer, file_load)
    # pnt.plot_loss(valid_loss, 'ReLu', optim_str, LR, model_type,
    #               'Loss dCt/dWk', epochs)
    # pnt.plot_ndcg(ndcg_list, 'ReLu', optim_str, LR, model_type, epochs)
    # # pnt.plot_earlystop(train_loss, valid_loss, model_type)
    # pr.plot_ARR(average_relevant_rank, 'ReLU', optim_str, LR, model_type, epochs)
    # # # get test scores from trained model
    # test_scores = model(data.test)
    # print('------')
    # print('Evaluation on entire test partition.')
    # # get results from test dataset and print results
    # results = eval.evaluate(data.test, test_scores, print_results=True)
    # filename = 'results_best_models/{}.json'.format(model_type)
    # pnt.save_to_json(filename, results)
