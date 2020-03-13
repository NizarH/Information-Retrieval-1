# import needed modules
import dataset
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
DIM_HIDDEN = 150  # Dimensions of the hidden layer
DIM_OUTPUT = 5  # Dimension of the output layer (number of labels)
LR = 0.001  # Learning rate
BATCH_SIZE = 32  # Batch size
NUM_EPOCHS = 2  # Number of epochs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        X = self.fc1(x)
        X = F.relu(X)
        X = self.fc2(X)
        return X


def model_train(model, data, optimizer, criterion):
    model.train()
    loss_list = []
    for epoch in range(NUM_EPOCHS):
        for i in tqdm(range(0, len(data.train.feature_matrix), BATCH_SIZE), position=0, leave=True):
            X = Variable(
                torch.tensor(data.train.feature_matrix[i: i + BATCH_SIZE], dtype=torch.float, requires_grad=False))  # gets input
            Y = Variable(
                torch.tensor(data.train.label_vector[i: i + BATCH_SIZE], requires_grad=False))  # gets correct output
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()
            F.softmax(y_pred, dim=1)
        loss_list.append(loss)
        print("Output: ", y_pred)
        print("Epoch {} - loss: {}".format(epoch, loss.item()))
    return model, optimizer, loss, loss_list


def save_model(model, epochs, optimizer, loss, lr, activ, optim):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'models/pointwise_ltr_eps={}_LR={}_{}_{}.pth.tar'.format(epochs, lr, activ, optim))


def load_model(model, optimizer):
    checkpoint = torch.load('models/pointwise_ltr.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return model, optimizer, checkpoint, epochs, loss


def plot_loss(loss, epochs, batch, activ, optim, lr):
    plt.plot(loss)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss for {} epochs'.format(epochs))
    plt.savefig('plots/loss_pointwise_epochs={}_batch={}_LR={}_{}_{}.png'.format(epochs, batch, lr, activ, optim))
    plt.show()


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    net = Net(data.num_features, DIM_HIDDEN, DIM_OUTPUT)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.4)
    # optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    model, optimizer, loss, loss_list = model_train(net, data, optimizer, criterion)
    plot_loss(loss_list, NUM_EPOCHS, BATCH_SIZE, 'ReLu', 'SGD', LR)
    # save_model(model, NUM_EPOCHS, optimizer, loss, LR, 'ReLu', 'Adam')
    # model, optimizer, checkpoint, epochs, loss = load_model(net, optimizer)
    # print('Loss: ', loss)