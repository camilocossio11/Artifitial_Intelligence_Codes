# %% LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import mpimg
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from customDataset import NumbersDataset
from pathlib import Path
# from load import load_all
from PIL import Image as im
sys.path.append(str(Path(__file__).parent / "ourMNIST"))

# %% FUNCTIONS


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120,
                      kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


# %%
def run():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # parameters
    RANDOM_SEED = 42
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    N_EPOCHS = 25
    IMG_SIZE = 32
    N_CLASSES = 10
    root = Path(os.getcwd())
    # DATA
    # image_dir = root/'processed_images'
    image_dir = root/'data_images'
    csv_file = root/'test_labels.csv'
    transform_img = transforms.Compose([transforms.Resize(32),
                                        # transforms.CenterCrop(80),
                                        transforms.ToTensor()
                                        ])
    dset = NumbersDataset(root, image_dir, csv_file, transform=transform_img)
    train_dataset, test_dataset = torch.utils.data.random_split(dset, [
                                                                186, 124])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=test_dataset,
                              batch_size=32, shuffle=True)
    # PLOTTING IMAGES
    ROW_IMG = 10
    N_ROWS = 5
    fig = plt.figure()
    for index in range(1, ROW_IMG * N_ROWS + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')
        plt.imshow(train_dataset[index][0][0], cmap='gray_r')
    fig.suptitle('Dataset - preview')

    torch.manual_seed(RANDOM_SEED)
    model = LeNet5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, x = training_loop(
        model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

    ROW_IMG = 10
    N_ROWS = 5

    fig2 = plt.figure()
    for index in range(1, ROW_IMG * N_ROWS + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')
        plt.imshow(test_dataset[index][0][0], cmap='gray_r')

        with torch.no_grad():
            model.eval()
            _, probs = model(test_dataset[index][0].unsqueeze(0))

        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'

        plt.title(title, fontsize=7)
    fig2.suptitle('LeNet-5 - predictions')


# %%
if __name__ == '__main__':
    run()
# %%
