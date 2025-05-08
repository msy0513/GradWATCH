import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import json

class LearnableFeatureGenerator(nn.Module):
    def __init__(self, embedding_dim, position_dim, sigma=0.1):
        super(LearnableFeatureGenerator, self).__init__()
        self.method_embedding = nn.Embedding(2, embedding_dim)
        self.value_linear = nn.Linear(1, embedding_dim)
        self.position_linear = nn.Linear(1, position_dim)
        self.time_linear = nn.Linear(1, embedding_dim)
        self.noise = np.random.normal(0, sigma, size=embedding_dim*2)
        self.embedding_dim = embedding_dim

    def forward(self, method_code, value, position, time_seconds):
        method_embed = self.method_embedding(method_code.long())
        value_embed = self.value_linear(value.view(-1, 1))
        position_embed = self.position_linear(position.float().view(-1, 1))

        time_embed = torch.tensor(
            time_encoding(time_seconds, d=self.embedding_dim), dtype=torch.float32, device=time_seconds.device
        )

        noise_tensor = torch.tensor(self.noise, dtype=torch.float32, device=method_embed.device)
        combined = torch.cat([method_embed, value_embed], dim=-1)
        combined_with_noise = torch.cat([combined + noise_tensor, position_embed, time_embed], dim=-1)

        return combined_with_noise

def time_encoding(t, d=16):
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    encoding = []
    for i in range(d):
        if i % 2 == 0:
            encoding.append(np.cos(t / 10000 ** (i / d)))
        else:
            encoding.append(np.sin(t / 10000 ** (i / d)))
    return np.array(encoding).T

def generate_node_features(addr2id_path, csv_files, embedding_dim=128, position_dim=16, save_dir="./node_feature/"):
    # 读取 addr2id（确保节点顺序一致）
    with open(addr2id_path, "r") as f:
        addr2id = json.load(f)

    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list)

    df['method_code'] = df['Method'].apply(lambda x: 1 if x.lower() == 'deposit' else 0)
    df['time_seconds'] = (pd.to_datetime(df['DateTime']) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df['position_index'] = df.index

    df['value_scaled'] = (df['Value_IN'].fillna(0) - df['Value_IN'].mean()) / df['Value_IN'].std()
    df['position_scaled'] = (df['position_index'] - df['position_index'].mean()) / df['position_index'].std()
    df['time_scaled'] = (df['time_seconds'] - df['time_seconds'].mean()) / df['time_seconds'].std()

    method_codes = torch.tensor(df['method_code'].values, dtype=torch.float32)
    values = torch.tensor(df['value_scaled'].values, dtype=torch.float32)
    positions = torch.tensor(df['position_scaled'].values, dtype=torch.float32)
    time_seconds = torch.tensor(df['time_scaled'].values, dtype=torch.float32)

    feature_generator = LearnableFeatureGenerator(embedding_dim, position_dim)
    optimizer = optim.Adam(feature_generator.parameters(), lr=1e-3)

    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        embeddings = feature_generator(method_codes, values, positions, time_seconds)
        loss = embeddings.norm()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    df['transaction_embedding'] = embeddings.detach().numpy().tolist()
    user_embeddings = df.groupby('From')['transaction_embedding'].apply(lambda x: np.mean(np.stack(x), axis=0))
    user_embeddings = user_embeddings.reset_index()
    user_embeddings.columns = ['User', 'User_Embedding']

    node_dim = embedding_dim * 2 + position_dim + embedding_dim
    node_feature = np.zeros((len(addr2id), node_dim))

    node2fea = {row.User: np.array(row.User_Embedding) for row in user_embeddings.itertuples()}

    for addr, idx in addr2id.items():
        node_feature[idx] = node2fea.get(addr, np.random.randn(node_dim))

    for i in range(100):
        np.save(f'{save_dir}{i}.npy', node_feature)

    print("Node features have been saved.")

if __name__ == "__main__":
    # addr2id_path = sys.argv[1]
    # csv_files = sys.argv[2:]
    # generate_node_features(addr2id_path, csv_files)

    addr2id_path = "../dataset/tornado-rule/addr2id.json"
    csv_files = [
        "../dataset/Concat/0.1ETH-concat.csv",
        "../dataset/Concat/1ETH-concat.csv",
        "../dataset/Concat/10ETH-concat.csv",
        "../dataset/Concat/100ETH-concat.csv"
    ]
    generate_node_features(addr2id_path, csv_files)
