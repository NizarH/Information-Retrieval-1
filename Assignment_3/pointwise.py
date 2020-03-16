# import needed modules
import dataset
import evaluate as eval
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import copy
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(5)
torch.manual_seed(0)
# Sets hyper-parameters
DIM_HIDDEN = 256  # Dimensions of the hidden layer
DIM_OUTPUT = 5  # Dimension of the output layer (number of labels)
NUM_LAYERS = 2
LR = 0.001  # Learning rate
NUM_EPOCHS = 100  # Number of epochs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        """
        Neural net with number of layers defined by user input. Uses dropout to prevent overfitting and relu as activation.
        """
        super(Net, self).__init__()
        self.layers = nn.ModuleList()   # input layer
        for layer in range(num_layers):     # add as many layers as num_layers
            if layer != 0:
                input_size = copy.copy(hidden_size)
                hidden_size = input_size // 2
            self.layers.append(nn.Sequential(   # make sequential of linear, dropout, relu
                nn.Linear(input_size, hidden_size),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
        self.layers.append(nn.Linear(hidden_size, output_size))     # output layer
        self.name = 'Pointwise LTR Model'

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def softmax_highest_score(output):
    """
    Function that converts output into probabilities and chooses the score with highest probability
    """
    # get probabilities from scores
    probs_scores = F.softmax(output, dim=1)
    # get index with highest probability
    highest = []
    for score_prob, score_output in zip(probs_scores, output):
        np_score_prob = score_prob.detach().squeeze().tolist()    # convert to python list
        np_score_output = score_output.detach().squeeze().tolist()    # convert to python list
        index_high_score = np_score_prob.index(max(np_score_prob))  # index highest probability
        high_score = np_score_output[index_high_score]     # get value with highest probability
        highest.append(high_score)
    return np.array(highest)


def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        model.bias.data.zero_()


def model_train(model, data, optimizer, criterion):
    model.train()

    loss_list = []
    ndcg_list = []

    X = torch.tensor(data.train.feature_matrix, dtype=torch.float, requires_grad=False)  # gets input
    Y = torch.tensor(data.train.label_vector, requires_grad=False)  # gets correct output
    validation_data = torch.tensor(data.validation.feature_matrix, dtype=torch.float, requires_grad=False)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()   # set gradients to zero
        y_pred = model(X)   # predict labels
        loss = criterion(y_pred, Y)     # calculate loss
        loss.backward()     # backpropagate loss
        optimizer.step()    # update weights
        print("Epoch {} - loss: {}".format(epoch, loss.item()))     # print loss

        if epoch % 5 == 0:  # print performance of model on validation data
            loss_list.append(loss)  # append loss to list to plot
            print("validation ndcg at epoch " + str(epoch))
            model.eval()
            validation_y_pred = model(validation_data)
            validation_scores = softmax_highest_score(validation_y_pred)
            results = eval.evaluate(data.validation, validation_scores, print_results=True)
            ndcg_list.append(results["ndcg"][0])
    return model, optimizer, loss, loss_list, ndcg_list


def save_model(model, epochs, optimizer, loss, lr, activ, optim):
    # save model, optimizer, loss to tar file
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'models/pointwise_ltr_eps={}_LR={}_{}_{}.pth.tar'.format(epochs, lr, activ, optim))


def load_model(model, optimizer):
    # load model from tar file
    checkpoint = torch.load('models/pointwise_ltr.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return model, optimizer, checkpoint, epochs, loss


def plot_loss(loss, epochs, activ, optim, lr):
    # plot cross entropy loss in graph
    plt.plot(loss, label='CE loss', color='blue')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss for {} epochs'.format(epochs))
    plt.savefig('plots/loss_pointwise_epochs={}_LR={}_{}_{}.png'.format(epochs, lr, activ, optim))
    plt.show()


def plot_ndcg(ndcg, epochs, activ, optim, lr):
    # plot ndcg in graph
    plt.plot(ndcg, label='NDCG', color='orange')
    plt.xlabel('Number of epochs')
    plt.ylabel('NDCG')
    plt.title('NDCG for {} epochs'.format(epochs))
    plt.savefig('plots/ndcg_pointwise_epochs={}_LR={}_{}_{}.png'.format(epochs, lr, activ, optim))
    plt.show()


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    # initialize model
    net = Net(data.num_features, DIM_HIDDEN, DIM_OUTPUT, NUM_LAYERS)
    net.apply(weights_init)
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    optim_str = optimizer.__str__()[0:3]
    # define loss function
    criterion = nn.CrossEntropyLoss()
    model, optimizer, loss, loss_list, ndcg_list = model_train(net, data, optimizer, criterion)
    plot_loss(loss_list, NUM_EPOCHS, 'ReLu', optim_str, LR)
    plot_ndcg(ndcg_list, NUM_EPOCHS, 'ReLu', optim_str, LR)
    save_model(model, NUM_EPOCHS, optimizer, loss, LR, 'ReLu', optim_str)
    # model, optimizer, checkpoint, epochs, loss = load_model(net, optimizer)
