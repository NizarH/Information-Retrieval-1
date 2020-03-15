# import needed modules
import dataset
import evaluate as eval
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(5)
torch.manual_seed(0)
# Sets hyper-parameters
DIM_HIDDEN = 256  # Dimensions of the hidden layer
DIM_OUTPUT = 5  # Dimension of the output layer (number of labels)
NUM_LAYERS = 1
LR = 0.001  # Learning rate
BATCH_SIZE = 32  # Batch size
NUM_EPOCHS = 10  # Number of epochs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])   # input layer
        for layer in range(num_layers):     # add as many layers as num_layers
            self.layers.append(nn.Sequential(   # make sequential of linear, dropout, relu
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
            hidden_size = hidden_size // 2  # divide hidden nodes by 2
        self.layers.append(nn.Linear(hidden_size, output_size))     # output layer
        self.name = 'Pointwise LTR Model'

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def output_softmax(self, output):
        return F.softmax(output, dim=1)     # convert output to probabilities with softmax

    def highest_score(self, output):
        # get index with highest probability
        highest = []
        for score in output:
            np_score = score.detach().squeeze().tolist()    # convert to python list
            print(np_score)
            max_score = np_score[np_score.index(max(np_score))]     # get highest probability
            highest.append(max_score)
        print(highest)
        return torch.tensor(highest)


def model_train(model, data, optimizer, criterion):
    model.train()
    loss_list = []
    for epoch in range(NUM_EPOCHS):
        for i in tqdm(range(0, len(data.train.feature_matrix), BATCH_SIZE), position=0, leave=True):
            X = Variable(
                torch.tensor(data.train.feature_matrix[i: i + BATCH_SIZE], dtype=torch.float, requires_grad=False))  # gets input
            Y = Variable(
                torch.tensor(data.train.label_vector[i: i + BATCH_SIZE], requires_grad=False))  # gets correct output
            optimizer.zero_grad()   # set gradients to zero
            y_pred = model(X)   # predict labels
            loss = criterion(y_pred, Y)     # calculate loss
            loss.backward()     # backpropagate loss
            optimizer.step()    # update weights
        loss_list.append(loss)      # append loss to list to plot
        print("Output: ", y_pred)   # print predicted output
        print("Sum output: ", np.sum(y_pred.detach().numpy()))
        print("Epoch {} - loss: {}".format(epoch, loss.item()))     # print loss
        
        ### evaluation of scores doesnt work yet
        # if epoch % 5 == 0:
        # print("validation ndcg at epoch " + str(epoch))
        # model.eval()
        # validation_data = Variable(torch.tensor(data.validation.feature_matrix, dtype=torch.float, requires_grad=False))
        # validation_y_pred = model(validation_data)
        # validation_scores = model.highest_score(model.output_softmax(validation_y_pred))
        # results = eval.evaluate(data.validation, validation_scores, print_results=False)
        # print(results["ndcg"])
    return model, optimizer, loss, loss_list


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


def plot_loss(loss, epochs, batch, activ, optim, lr):
    # plot cross entropy loss in graph
    plt.plot(loss)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss for {} epochs'.format(epochs))
    plt.savefig('plots/loss_pointwise_epochs={}_batch={}_LR={}_{}_{}.png'.format(epochs, batch, lr, activ, optim))
    plt.show()


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    # initialize model
    net = Net(data.num_features, DIM_HIDDEN, DIM_OUTPUT, NUM_LAYERS)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.5)
    # optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    optim_str = optimizer.__str__()[0:3]
    criterion = nn.CrossEntropyLoss()
    model, optimizer, loss, loss_list = model_train(net, data, optimizer, criterion)
    plot_loss(loss_list, NUM_EPOCHS, BATCH_SIZE, 'ReLu', optim_str, LR)
    # save_model(model, NUM_EPOCHS, optimizer, loss, LR, 'ReLu', 'Adam')
    # model, optimizer, checkpoint, epochs, loss = load_model(net, optimizer)
