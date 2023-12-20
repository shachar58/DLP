import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Autoencoder(nn.Module):

    def __init__(self, input_size, embedding_size=16):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, data):
        data = self.encoder(data)
        data = self.decoder(data)
        return data

    def get_embedding(self, data):
        new_sample_tensor = data.clone().detach().unsqueeze(0)
        with torch.no_grad():
            embedding = self.encoder(new_sample_tensor)
        return embedding

    def create_df_vec(self, tensor_data):
        embed_l = []
        for line in tensor_data:
            embed_l.append(self.get_embedding(line).squeeze())
        vectors_array = np.stack([tensor.cpu().numpy() for tensor in embed_l])
        df_vec = pd.DataFrame(vectors_array)
        return df_vec


def train(input_size, tensor_data, dataset_name, lr=0.000032, epochs=35):
    autoencoder = Autoencoder(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    X_train = tensor_data  # Replace with your actual data
    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    num_epochs = epochs
    epoch_losses = []  # List to store loss values
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            reconstructed = Autoencoder(inputs)
            loss = criterion(reconstructed, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Total Loss {total_loss:.4f}')

    model_file_name = f"/content/drive/MyDrive/model/{dataset_name}.mdl"
    # model_file_name = f"/content/drive/MyDrive/model/combined benign-5.mdl"
    torch.save(autoencoder.state_dict(), model_file_name)
    print(f"Saving AutoEncoder: {model_file_name}")

    # Plotting the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', linestyle='-')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    return autoencoder


def load(input_size, ae_model_to_load):
    autoencoder = Autoencoder(input_size)
    print("Loading AutoEncoder" + " " + ae_model_to_load)
    autoencoder.load_state_dict(torch.load(ae_model_to_load))
    autoencoder.eval()
    return autoencoder


def get_embedding(tensor, autoencoder):
    new_sample_tensor = tensor.clone().detach().unsqueeze(0)
    with torch.no_grad():
        embedding = autoencoder.encoder(new_sample_tensor)
    return embedding


def create_vec_df(tensor_data, autoencoder):
    embed_l = []
    for x in tensor_data:
        embed_l.append(get_embedding(x, autoencoder).squeeze())
    vectors_array = np.stack([tensor.cpu().numpy() for tensor in embed_l])
    df_vec = pd.DataFrame(vectors_array)
    return df_vec, vectors_array
