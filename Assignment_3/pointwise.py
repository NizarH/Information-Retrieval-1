# import needed modules
import dataset
import evaluate as eval
import pairwise as pr
import pairwise_spedup as sped
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from pytorchtools import EarlyStopping
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import operator

# fix random seed for reproducibility
np.random.seed(5)
torch.manual_seed(0)
# Sets hyper-parameters
DIM_HIDDEN = 128  # Dimensions of the hidden layer
DIM_OUTPUT = 5  # Dimension of the output layer (number of labels)
NUM_LAYERS = 4
LR = 0.0001  # Learning rate
MOMENTUM = 0.3  # Momentum for SGD
DROPOUT = 0.3  # Dropout
NUM_EPOCHS = 100  # Number of epochs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        """
        Neural net with number of layers defined by user input. Uses dropout to prevent overfitting and relu as activation.
        """
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.check_num_layers()
        self.layers = nn.ModuleList()  # input layer
        for layer in range(self.num_layers):  # add as many layers as num_layers
            if layer != 0:
                input_size = copy.copy(self.hidden_size)
                self.hidden_size = input_size // 2
            self.layers.append(nn.Sequential(  # make sequential of linear, dropout, relu
                nn.Linear(input_size, self.hidden_size),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))  # output layer
        self.name = 'Pointwise LTR Model'

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def check_num_layers(self):
        max_num_layers = 0
        new_hidden = self.hidden_size
        for i in range(self.num_layers):
            div_hidden = new_hidden // 2
            if div_hidden > self.output_size:
                max_num_layers += 1
            new_hidden = div_hidden
        if self.num_layers > max_num_layers:
            print('Number of layers too large. Changed to {} layers'.format(max_num_layers))
            self.num_layers = max_num_layers


def softmax_highest_score(output):
    """
    Function that converts output into probabilities and chooses the score with highest probability
    """
    # get probabilities from scores
    probs_scores = F.softmax(output, dim=1)
    # get index with highest probability
    highest = []
    highest_index = []
    for score_prob, score_output in zip(probs_scores, output):
        np_score_prob = score_prob.detach().squeeze().tolist()  # convert to python list
        np_score_output = score_output.detach().squeeze().tolist()  # convert to python list
        index_high_score = np_score_prob.index(max(np_score_prob))  # index highest probability
        high_score = np_score_output[index_high_score]  # get value with highest probability
        highest.append(high_score)
        highest_index.append(index_high_score)
    return np.array(highest), np.array(highest_index)


def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        model.bias.data.zero_()


def model_train(model, data, optimizer, criterion, epochs=NUM_EPOCHS, patience=20):
    model.train()
    model_type = 'pointwise'
    scores = []
    train_losses = []
    valid_losses = []
    valid_indexes = []
    ndcg_list = []

    X = torch.tensor(data.train.feature_matrix, dtype=torch.float, requires_grad=False)  # gets input
    Y = torch.tensor(data.train.label_vector, requires_grad=False)  # gets correct output
    validation_data = torch.tensor(data.validation.feature_matrix, dtype=torch.float,
                                   requires_grad=False)  # validation input
    validation_target = torch.tensor(data.validation.label_vector, requires_grad=False)  # validation correct output

    # initialize the early_stopping object
    early_stopping = EarlyStopping(model_type, patience=patience, verbose=True, delta=0.0001)

    for epoch in tqdm(range(epochs), position=0, leave=True):
        optimizer.zero_grad()  # set gradients to zero
        y_pred = model(X)  # predict labels
        loss = criterion(y_pred, Y)  # calculate loss
        loss.backward()  # backpropagate loss
        optimizer.step()  # update weights
        train_losses.append(loss.item())  # append loss to list to plot

        print("validation ndcg at epoch " + str(epoch))
        model.eval()
        validation_y_pred = model(validation_data)
        validation_scores, validation_indexes = softmax_highest_score(validation_y_pred)
        scores.append(validation_scores)
        # calculate the loss
        valid_loss = criterion(validation_y_pred, validation_target)
        # record validation loss
        valid_losses.append(valid_loss.item())
        valid_indexes.append(validation_indexes)
        results = eval.evaluate(data.validation, validation_scores, print_results=False)
        ndcg_list.append(results['ndcg']['mean'])
        # print('ndcg: ', results["ndcg"])
        print("Epoch {} - train loss: {} - validation loss: {}".format(epoch, loss.item(),
                                                                       valid_loss.item()))  # print loss

        if epoch % 5 == 0:  # print performance of model on validation data
            epoch_len = len(str(epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'train_loss: {loss.item():.5f} ' +
                         f'valid_loss: {valid_loss.item():.5f}')
            print(print_msg)

        # early_stopping checks if validation loss has decresed
        early_stopping(valid_loss.item(), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('models/{}_checkpoint.pt'.format(model_type)))

    return model, optimizer, scores, train_losses, valid_losses, ndcg_list, validation_indexes


def tune_params(data):
    ltr_type = 'pairwise_spedup'
    loss_fn = 'Loss dCt/dWk'
    epochs = 80
    numpairs = 2000
    LR = 0.01
    dropout = 0.1
    sigma = 0.1
    node_size = [10]
    # epochs = [100, 80, 50]
    best_str = 'number of layers'
    str_file = 'layers'
    # lrs = [0.01, 0.001, 0.0001, 0.00001]
    # num_nodes = [128, 64, 32, 16, 10]
    # num_layers = 5
    layers = [node_size*50, node_size*25, node_size*10, node_size*5, node_size*3]
    # nodes = [256, 128, 64, 32, 16]
    # momentums = np.random.uniform(0.1, 0.9, size=5)
    # sigmas = [0.1, 0.5, 1, 2, 5]
    # dropouts = [0.1, 0.2, 0.3, 0.4]
    # pairs = [5000, 2000, 1000, 500, 100]
    ndcg_scores = {}
    # best_params = {}
    for num_layers in layers:
        # model = Net(data.num_features, DIM_HIDDEN, DIM_OUTPUT, NUM_LAYERS)
        # model.apply(weights_init)
        model = sped.pairWiseModel(data.num_features, num_layers, dropout=dropout, sigma=sigma, pairs=numpairs)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        # model, optimizer, loss, train_loss, valid_loss, ndcg_list = model_train(model, data, optimizer, criterion)
        model, optimizer, train_losses, validation_losses, ndcg_list = sped.trainModel(model, data, epochs, optimizer,
                                                                                       ltr_type)
        optim_str = optimizer.__str__()[0:3]
        mean_ndcg = np.mean(ndcg_list)
        print('{}: '.format(best_str), len(num_layers), 'mean ndcg: ', mean_ndcg)
        ndcg_scores['{}={}'.format(str_file, len(num_layers))] = mean_ndcg
        plot_loss(validation_losses, '{}='.format(best_str) + str(len(num_layers)), 'ReLu', optim_str, LR, ltr_type, loss_fn,
                  epochs)
        plot_ndcg(ndcg_list, '{}='.format(best_str) + str(len(num_layers)), 'ReLu', optim_str, LR, ltr_type, epochs)

    best_key = max(ndcg_scores.items(), key=operator.itemgetter(1))[0]
    # best_params['Best {}'.format(best_str)] = best_key

    hyperparams = {'NDCG for {}'.format(best_str): ndcg_scores}
    filename = 'hyperparams_{}/{}_optim={}_activ=ReLU_loss=dCtdWk.json'.format(ltr_type, str_file, optim_str)
    save_to_json(filename, hyperparams)

    filename_best = 'hyperparams_{}/best_hyperparameters_optim={}_activ=ReLU_loss=dCtdWk.json'.format(ltr_type,
                                                                                                      optim_str)
    with open(filename_best, 'r') as f:
        best_params = json.load(f)
        best_params['Best {}'.format(best_str)] = best_key

    save_to_json(filename_best, best_params)


def save_model(model, epochs, optimizer, valid_scores, train_loss, valid_loss, ndcg_list, average_relevant_rank, lr, activ, optim, model_type):
    # save model, optimizer, loss to tar file
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_scores': valid_scores,
        'training_loss': train_loss,
        'validaton_loss': valid_loss,
        'ndcg_list': ndcg_list,
        'average_rel_rank': average_relevant_rank
    }, 'models/{}_ltr_tuned_eps={}_LR={}_{}_{}.pth.tar'.format(model_type, epochs, lr, activ, optim))


def load_model(model, optimizer, file):
    # load model from tar file
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    validation_scores = checkpoint['validation_scores']
    train_loss = checkpoint['training_loss']
    valid_loss = checkpoint['validaton_loss']
    ndcg = checkpoint['ndcg_list']
    average_relevant_rank = checkpoint['average_rel_rank']
    model.eval()
    return model, optimizer, checkpoint, epochs, validation_scores, train_loss, valid_loss, ndcg, average_relevant_rank


def save_model_pointwise(model, epochs, optimizer, valid_scores, train_loss, valid_loss, validation_indexes, ndcg_list, lr, activ, optim, model_type):
    # save model, optimizer, loss to tar file
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_scores': valid_scores,
        'training_loss': train_loss,
        'validaton_loss': valid_loss,
        'validation_index': validation_indexes,
        'ndcg_list': ndcg_list,
    }, 'models/{}_ltr_tuned_eps={}_LR={}_{}_{}.pth.tar'.format(model_type, epochs, lr, activ, optim))


def load_model_pointwise(model, optimizer, file):
    # load model from tar file
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    validation_scores = checkpoint['validation_scores']
    train_loss = checkpoint['training_loss']
    valid_loss = checkpoint['validaton_loss']
    # valid_index = checkpoint['validation_index']
    ndcg = checkpoint['ndcg_list']
    model.eval()
    return model, optimizer, checkpoint, epochs, validation_scores, train_loss, valid_loss, ndcg


def save_to_json(filename, data, indent=4):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=indent)


def plot_loss(loss, activ, optim, lr, model_type, loss_fn, epochs):
    # x_values = [0, 10, 20, 30, 40, 50, 60, 80]
    # plot cross entropy loss in graph
    # plt.plot(x_values, loss, color='blue')
    plt.plot(loss, label='CE loss', color='blue')
    plt.xlabel('Number of epochs')
    plt.ylabel(loss_fn)
    plt.title('Loss of {} model'.format(model_type))
    plt.savefig('plots/validationloss_{}_tuned_epochs={}_LR={}_{}_{}.png'.format(model_type, epochs, lr, activ, optim))
    plt.show()


def plot_ndcg(ndcg, activ, optim, lr, model_type, epochs):
    # x_values = [0, 10, 20, 30, 40, 50, 60, 80]
    # plot ndcg in graph
    # plt.plot(x_values, ndcg, label='NDCG', color='orange')
    plt.plot(ndcg, label='NDCG', color='orange')
    plt.xlabel('Number of epochs')
    plt.ylabel('NDCG')
    plt.title('NDCG scores of {} model'.format(model_type))
    plt.savefig('plots/validationndcg_{}_tuned_epochs={}_LR={}_{}_{}.png'.format(model_type, epochs, lr, activ, optim))
    plt.show()


def plot_earlystop(train_loss, valid_loss, model_type):
    # x_values = [0, 10, 20, 30, 40, 50, 60, 80]
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    # plt.plot(x_values, train_loss, label='Training Loss')
    # plt.plot(x_values, valid_loss, label='Validation Loss')
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.title('Training loss vs. Validation loss of {} model'.format(model_type))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.show()
    fig.savefig('plots/train_valid_loss_plot_{}.png'.format(model_type), bbox_inches='tight')


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    # initialize model
    net = Net(data.num_features, DIM_HIDDEN, DIM_OUTPUT, NUM_LAYERS)
    net.apply(weights_init)
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    optim_str = optimizer.__str__()[0:3]
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # tune_params(data)
    # model, optimizer, scores, train_losses, valid_losses, ndcg_list, validation_indexes = model_train(net, data, optimizer, criterion)
    model_type = 'pointwise'
    # plot_loss(valid_losses, 'ReLu', optim_str, LR, model_type, 'Cross Entropy Loss', NUM_EPOCHS)
    # plot_ndcg(ndcg_list, 'ReLu', optim_str, LR, model_type, NUM_EPOCHS)
    # plot_earlystop(train_losses, valid_losses, model_type)

    # save_model_pointwise(model, NUM_EPOCHS, optimizer, scores, train_losses, valid_losses, validation_indexes, ndcg_list, LR, 'ReLU', optim_str, model_type)
    file_load = 'models/pointwise_ltr_tuned_eps=100_LR=0.0001_ReLU_Ada.pth.tar'
    model, optimizer, checkpoint, epochs, validation_scores, train_loss, valid_loss, ndcg_list = load_model_pointwise(net, optimizer, file_load)
    plot_loss(valid_loss, 'ReLu', optim_str, LR, model_type, 'Cross Entropy Loss', NUM_EPOCHS)
    plot_ndcg(ndcg_list, 'ReLu', optim_str, LR, model_type, NUM_EPOCHS)
    plot_earlystop(train_loss, valid_loss, model_type)
    # plt.hist(np.array(valid_index), bins=30, color='purple')
    # plt.show()

    test_data = torch.tensor(data.test.feature_matrix, dtype=torch.float, requires_grad=False)  # test input
    test_target = torch.tensor(data.test.label_vector, requires_grad=False)  # test correct output
    plt.hist(np.array(test_target), bins=50, color='orange')
    plt.show()

    # get test scores from trained model
    test_y_pred = model(test_data)
    test_scores, test_indexes = softmax_highest_score(test_y_pred)
    plt.hist(np.array(test_indexes), bins=30, color='green')
    plt.show()
    print('--------')
    print('Evaluation on entire test partition.')
    # get results from test dataset and print results
    results = eval.evaluate(data.test, test_scores, print_results=True)
    filename = 'results_best_models/pointwise_eps={}.json'.format(NUM_EPOCHS)
    save_to_json(filename, results)
