# import needed modules
import dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

# Sets hyper-parameters
DIM_HIDDEN = 150  # Dimensions of the hidden layer
DIM_OUTPUT = 5  # Dimension of the output layer (number of labels)
LR = 0.001  # Learning rate
BATCH_SIZE = 10  # Batch size
NUM_EPOCHS = 10  # Number of epochs

# This part creates the dataframe
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()

# Creates the weights
W1 = Variable(torch.randn(data.num_features, DIM_HIDDEN).double(), requires_grad=True)  # weights from input to hidden
W2 = Variable(torch.randn(DIM_HIDDEN, DIM_OUTPUT).double(), requires_grad=True)  # weights from hidden to output

for epoch in range(NUM_EPOCHS):
    for i in tqdm(range(0, len(data.train.feature_matrix), BATCH_SIZE), position=0, leave=True):
        x = Variable(torch.tensor(data.train.feature_matrix[i: i + BATCH_SIZE], requires_grad=False))  # gets input
        y = Variable(torch.tensor(data.train.label_vector[i: i + BATCH_SIZE], requires_grad=False))  # gets correct output

        y_pred = x.mm(W1).clamp(min=0).mm(W2)  # multiplies the input with the weights of the layers
        criterion = nn.CrossEntropyLoss()  # Creates loss object
        loss = criterion(y_pred, y)  # Calculates loss
        loss.backward()  # Backpropegates

        W1.data -= LR * W1.grad.data  # Changes the weight according to the gradient
        W2.data -= LR * W2.grad.data  # Changes the weight according to the gradien

        W1.grad.data.zero_()  # Sets the gradient to zero for the next iteration
        W2.grad.data.zero_()  # Sets the gradient to zero for the next iteration
    print(f'The loss at epoch {epoch} is {loss.item()}')  # prints the loss at the end of every epoch
