import torch.nn as nn
import torch
import numpy as np
import pandas as pd


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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_embedding(x, autoencoder):
    new_sample_tensor = x.clone().detach().unsqueeze(0)
    with torch.no_grad():
        embedding = autoencoder.encoder(new_sample_tensor)
    return embedding


def create_df_vec(tensor_data, autoencoder):
    embed_l = []
    for x in tensor_data:
        embed_l.append(get_embedding(x, autoencoder).squeeze())
    vectors_array = np.stack([tensor.cpu().numpy() for tensor in embed_l])
    df_vec = pd.DataFrame(vectors_array)
