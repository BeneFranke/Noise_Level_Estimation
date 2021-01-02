import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import matplotlib.pyplot as plt

from statistics import mean
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit

# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, '\n', torch.cuda.get_device_name(0))

# working directory
path = "C:\\Python\\Thesis_Work"

# load features and labels
df = pd.read_csv(path + "\\Features\\MFCC_23.csv", sep=";")           # MFCCs
X = torch.tensor(df.iloc[:, 0:23].values)
Y = torch.tensor(df[['SPL_dBA']].values)
LABEL = np.array(df[['Label']])
GROUPS = np.array(df[['File_Name']])

# tunable global hyper parameters
N_FEATURES = int(np.array(X[0, :].shape))
N_HIDDEN = 4 * N_FEATURES
N_LAYERS = 4
N_BATCHES = 4000                 # -> len(data)/batch_size = number of batches
N_EPOCHS = 60
LEARNING_RATE = 0.001           # typically 0.01 or 0.001
DROPOUT = 0


# dataset for mfcc and spl
class AudioDataset(Dataset):
    def __init__(self, X, Y):#, label):
        # loading data
        self.X = X
        self.y = Y
        #self.label = label
        # get length and shape
        self.lenX = len(X)
        self.lenY = len(Y)
        self.shapeX = X.shape
        self.shapeY = Y.shape
        #self.shapeLabel = label.shape

    def __len__(self):
        return self.lenX

    def __getitem__(self, i):
        return self.X[i, :], self.y[i]#, self.label[i]


# long short-term memory
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # building LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=DROPOUT, batch_first=True)
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_len, feature_dim)
        # batch_first=False causes input/output tensors to be of shape (seq_len, batch_size, feature_dim)

        # building linear output layer
        self.lin_out = nn.Linear(self.hidden_size, out_features=1)

    def init_hidden(self):
        # initialize first hidden state h0 and first cell state c0
        # the dimension axes are (num_layers, batch_size, hidden_size)
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        # run lstm block
        lstm_out, hidden = self.lstm(x.view(x.shape[0], 1, N_FEATURES))
        out = self.lin_out(lstm_out[:, -1, :])                           # (batch_dim, seq_dim, feature_dim)
        return out


# split data in train & test set
def split_train_test_data(check_groups=False):
    # split data with Group Shuffle Split
    GSS = GroupShuffleSplit(test_size=0.2, n_splits=1)       # no n_splits
    for train_idx, test_idx in GSS.split(X, Y, GROUPS):  # groups parameter = file names
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        label_train, label_test = LABEL[train_idx], LABEL[test_idx]
        groups_train, groups_test = GROUPS[train_idx], GROUPS[test_idx]

    if check_groups:
        group_check(groups_train, groups_test, frame_size=1407)

    return X_train, y_train, label_train, groups_train, X_test, y_test, label_test, groups_test
    # return X_train, y_train, X_test, y_test


# check if data from one sample is either in test or in train data
def group_check(g_train, g_test, frame_size):
    er = False
    for i in range(0, len(g_train), frame_size):
        for j in range(0, len(g_test), frame_size):
            if g_train[i] == g_test[j]:
                print("ERROR", g_train[i], "=", g_test[j])
                er = True
                break

    if not er:
        print("-- Train & Test Data Check -- FINISHED -- EVERYTHING OK")


# train model
def train(train_loader, model, optimizer, criterion, n_epochs=5):
    cost = []
    metric_hist = []
    for epoch in range(n_epochs):
        total_loss = total_metrics = 0
        for i, (x, y) in enumerate(train_loader):

            x = x.float().to(device)
            y = y.float().to(device)

            model.zero_grad()

            # forward
            y_pred = model(x)

            # calc loss & rmse
            loss = criterion(y_pred, y)                                     # LOSS = Mean Square Error
            metrics = rmse(y_pred, y)                                       # METRICS = Root Mean Square Error

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # append results
            total_loss += loss.item()
            total_metrics += metrics.item()

            # print status every 1000th step
            if (i+1) % 1000 == 0:
                print(i+1, " out of ", len(train_loader), "iterations: ",
                      round(((i+1) / len(train_loader)) * 100, 2), "% Done -- LOSS: ", round(loss.item(), 3),
                      "-- RMSE: ", round(metrics.item(), 3))

        # calc average loss of epoch and append to cost
        avg_metrics = total_metrics / len(train_loader)
        avg_loss = total_loss / len(train_loader)
        cost.append(avg_loss)
        metric_hist.append(avg_metrics)

        print(epoch+1, " out of ", n_epochs, "epochs: ", round(((epoch+1) / n_epochs) * 100, 3)
              , "% Done --", "LOSS: ", round(avg_loss, 2), "-- RMSE: ", round(avg_metrics, 3), '\n')

    # return cost and metric over all epochs
    return cost, metric_hist


# test model
def test(test_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        rmse_hist = []
        total_loss = total_metrics = 0
        for i, (x, y) in enumerate(test_loader):

            x = x.float().to(device)
            y = y.float().to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            metrics = rmse(y_pred, y)

            total_loss += loss.item()
            total_metrics += metrics.item()
            rmse_hist.append([np.asarray(y.view([]).to("cpu")),
                              np.asarray(y_pred.view([]).to("cpu")),
                              np.asarray(round(metrics.item(), 2))])

            # print status every 1000th step
            if (i + 1) % 1000 == 0:
                print(i+1, " out of ", len(test_loader), "iterations: ",
                    round(((i+1) / len(test_loader)) * 100, 2), "% Done -- LOSS: ", round(loss.item(), 2),
                    "-- RMSE: ", round(metrics.item(), 2))

    # average loss and rmse over train interval
    avg_loss = round(total_loss/len(test_loader), 2)
    avg_metrics = round(total_metrics/len(test_loader), 2)

    return avg_loss, avg_metrics, rmse_hist

# calc RMSE
def rmse(y_pred, y_true):
    return torch.mean(torch.sqrt((y_pred - y_true) ** 2))


if __name__ == "__main__":
    # load train and test data -> [X, y, class, file name]
    X_train, y_train, label_train, name_train, X_test, y_test, label_test, name_test = split_train_test_data()

    # define batch size
    batch_size_train = int(len(X_train) / N_BATCHES)
    batch_size_test = 1

    # initialize model
    model = LSTM(input_size=N_FEATURES, hidden_size=N_HIDDEN, num_layers=N_LAYERS,
                 batch_size=int(len(X_train)/N_BATCHES))
    model = model.to(device)

    # define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # create Datasets and Dataloader
    train_data = AudioDataset(X=X_train, Y=y_train)#
    test_data = AudioDataset(X=X_test, Y=y_test)

    print("Length Train Data: ", len(train_data))
    print("Length Test Data: ", len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size_test, shuffle=False)

    examples = iter(train_loader)                     # iter through train_loader with batch size and seq_len iteration
    mfcc, spl = examples.next()                       # gives next batch-load
    print("Shape of 1 Iter (X, y): ", mfcc.shape, spl.shape)      # gives batch size x features / labels

    print('\n', model, '\n')

    # enable anomaly detection for training loop
    torch.autograd.set_detect_anomaly(True)

    # start timer
    start = timeit.default_timer()

    # training loop
    cost, train_rmse = train(train_loader=train_loader, model=model,
                             criterion=criterion, optimizer=optimizer, n_epochs=N_EPOCHS)

    # print training time
    stop_train = timeit.default_timer()
    print('Training Time: ', round(stop_train - start, 3), 's', '\n')

    # print training results
    print("Results after Training")
    print("----------")
    print("Min Loss:", round(min(cost), 2))
    print("Avg Loss:", round(mean(cost), 2))
    print("Min RMSE:", round(min(train_rmse), 2))
    print("Avg RMSE:", round(mean(train_rmse), 2), '\n')

    # plot cost function log
    y = cost
    x = np.arange(N_EPOCHS)+1
    plt.plot(x, y, '-r')
    plt.xlim((1, N_EPOCHS+1))
    plt.yscale("log")
    plt.title('Cost function of LSTM model')
    plt.xlabel('Epoch')
    plt.ylabel('MSE [dB]')
    plt.grid(True)
    plt.savefig(path + "\\Plots\\error_func\\ef_lstm_log")
    plt.close()

    # plot cost function linear
    y = cost
    x = np.arange(N_EPOCHS)+1
    plt.plot(x, y, '-r')
    plt.xlim((1, N_EPOCHS+1))
    plt.title('Cost function of LSTM model')
    plt.xlabel('Epoch')
    plt.ylabel('MSE [dB]')
    plt.grid(True)
    plt.savefig(path + "\\Plots\\error_func\\ef_lstm")
    plt.show()

    # testing loop
    test_loss, test_rmse, test_rmse_hist = test(test_loader=test_loader, model=model, criterion=criterion)

    # print testing results
    print("\nResults After Testing")
    print("---------------------")
    print("Average LOSS: ", test_loss)
    print("Average RMSE: ", test_rmse, '\n')

    # print testing time
    stop_test = timeit.default_timer()
    print('Testing Time: ', round(stop_test - stop_train - start, 3), 's', '\n')

    test_rmse_hist = np.asarray(test_rmse_hist)
    test_results = np.c_[test_rmse_hist, label_test, name_test]

    # save detailed testing results for any sample
    df = pd.DataFrame(data=test_results, columns=["y_true", "y_pred", "y_dif", "class", "recording"])
    df.to_csv(path+"\\Results\\LSTM_"+str(N_FEATURES)+"features_"+str(N_EPOCHS)+"epochs.csv"
              , index=False, sep=";")

    # save training & evaluation error to file
    df = pd.DataFrame(data=np.c_[round(train_rmse[-1], 2), test_rmse], columns=["Training Error", "Evaluation Error"])
    df.to_csv(path+"\\Results\\RMSE\\LSTM"+str(N_FEATURES)+"features_"+str(N_EPOCHS)+"epochs.csv", sep=";")

    # save model
    torch.save(model, open(path + "\\Models\\LSTM.sav", 'wb'))
