import json
from model import NeuralNet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# x_data este bag of words, y_data este indicele tagului si tags sunt numele tagurilor

def train_NN(x_data, y_data):
    class SentenceDataset(Dataset):
        def __init__(self):
            self.data_size = len(x_data)
            self.x_data = x_data
            self.y_data = y_data

        def __len__(self):
            return self.data_size

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

    input_layer_size = len(x_data[0])
    hidden_layer_size = 30
    output_layer_size = len(y_data)
    learning_rate = 0.1
    epochs_number = 1000
    batch_size = 10

    dataset = SentenceDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = NeuralNet(input_layer_size, hidden_layer_size, output_layer_size)

    # loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs_number):
        epoch_max_loss = 0
        for (sentences, labels) in data_loader:
            labels = labels.long()

            output_labels = model(sentences)

            loss = loss_criterion(output_labels, labels)
            loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()
        print(epoch)
        if (epoch + 1) % 100 == 0:
            print(f' loss for epoch {epoch + 1} is: {loss:.8f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_layer_size,
        "output_size": output_layer_size,
        "hidden_size": hidden_layer_size
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training ended, the data was saved to {FILE}')