import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Autoencoder(nn.Module):
    """
  A simple autoencoder class that consists of an encoder and a decoder.

  Attributes:
      encoder (Sequential): A sequence of layers that encode the input data.
      decoder (Sequential): A sequence of layers that decode the encoded data.

  Methods:
      forward(data): Passes data through the encoder and decoder.
      get_embedding(data): Extracts the embedding (encoded data) for a given input.
      create_df_vec(tensor_data): Creates a DataFrame of embeddings for a set of input tensors.
  """
    def __init__(self, input_size, embedding_size=16):
        """
        Initializes the autoencoder with a specified input size and embedding size.
        Parameters:
            input_size (int): The size of the input data.
            embedding_size (int): The size of the embedding (encoded data).
        """
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
        """
        Passes data through the autoencoder (encoder and decoder).
        Parameters:
            data (Tensor): The input data.
        Returns:
            Tensor: The reconstructed data after encoding and decoding.
        """
        data = self.encoder(data)
        data = self.decoder(data)
        return data

    def get_embedding(self, data):
        """
       Extracts the embedding (encoded data) from the input data.
       Parameters:
           data (Tensor): The input data.
       Returns:
           Tensor: The embedding (encoded data).
       """
        new_sample_tensor = data.clone().detach().unsqueeze(0)
        with torch.no_grad():
            embedding = self.encoder(new_sample_tensor)
        return embedding

    def create_df_vec(self, tensor_data):
        """
        Creates a DataFrame of embeddings for a given set of input tensors.
        Parameters:
            tensor_data (Tensor): A set of input tensors.
        Returns:
            DataFrame: A DataFrame containing the embeddings for each input tensor.
        """
        embed_l = []
        for line in tensor_data:
            embed_l.append(self.get_embedding(line).squeeze())
        vectors_array = np.stack([tensor.cpu().numpy() for tensor in embed_l])
        df_vec = pd.DataFrame(vectors_array)
        return df_vec


def train(input_size, tensor_data, dataset_name, lr=0.00005, epochs=35):
    """
    Trains the autoencoder model on the provided data.
    Parameters:
        input_size (int): The size of the input data.
        tensor_data (Tensor): The training data.
        dataset_name (str): The name of the dataset, used for saving the model.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
    Returns:
        Autoencoder: The trained autoencoder model.
    """
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
            reconstructed = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Total Loss {total_loss:.4f}')

    model_file_name = f"models/{dataset_name}.mdl"
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
    """
    Loads a pre-trained autoencoder model from a file.
    Parameters:
        input_size (int): The size of the input data.
        ae_model_to_load (str): File path to the saved autoencoder model.
    Returns:
        Autoencoder: The loaded autoencoder model.
    """
    autoencoder = Autoencoder(input_size)
    print("Loading AutoEncoder" + " " + ae_model_to_load)
    autoencoder.load_state_dict(torch.load(ae_model_to_load))
    autoencoder.eval()
    return autoencoder


def get_embedding(tensor, autoencoder):
    """
    Extracts the embedding for a single tensor using the provided autoencoder.
    Parameters:
        tensor (Tensor): The input tensor.
        autoencoder (Autoencoder): The autoencoder model.
    Returns:
        Tensor: The embedding (encoded data).
    """
    new_sample_tensor = tensor.clone().detach().unsqueeze(0)
    with torch.no_grad():
        embedding = autoencoder.encoder(new_sample_tensor)
    return embedding


def create_vec_df(tensor_data, autoencoder):
    """
    Creates a DataFrame and an array of embeddings for a set of input tensors.
    Parameters:
        tensor_data (Tensor): A set of input tensors.
        autoencoder (Autoencoder): The autoencoder model.
    Returns:
        tuple: A tuple containing a DataFrame of embeddings and an array of embeddings.
    """
    embed_l = []
    for x in tensor_data:
        embed_l.append(get_embedding(x, autoencoder).squeeze())
    vectors_array = np.stack([tensor.cpu().numpy() for tensor in embed_l])
    df_vec = pd.DataFrame(vectors_array)
    return df_vec, vectors_array
