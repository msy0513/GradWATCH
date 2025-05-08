import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import MinMaxScaler
from nodedemo.input_embedding4 import LearnableFeatureGenerator
import json
import subprocess


# df_2 = pd.read_csv(f'./dataset/Concat-gas/1ETH-concat-gas.csv', usecols=['UnixTimestamp', 'From', 'To'])
df_1 = pd.read_csv(f'./dataset/Concat/0.1ETH-concat.csv', usecols=['UnixTimestamp', 'From', 'To', 'Value_IN', 'Value_OUT','Method','GasPrice'])
df_2 = pd.read_csv(f'./dataset/Concat/1ETH-concat.csv', usecols=['UnixTimestamp', 'From', 'To', 'Value_IN', 'Value_OUT','Method','GasPrice'])
df_3 = pd.read_csv(f'./dataset/Concat/10ETH-concat.csv', usecols=['UnixTimestamp', 'From', 'To', 'Value_IN', 'Value_OUT','Method','GasPrice'])
df_4 = pd.read_csv(f'./dataset/Concat/100ETH-concat.csv', usecols=['UnixTimestamp', 'From', 'To', 'Value_IN', 'Value_OUT','Method','GasPrice'])
link_df = pd.concat([df_1, df_2, df_3, df_4])
link_df.rename(columns={'UnixTimestamp': 'timestamp', 'From':'sender', 'To':'receiver'}, inplace=True)
# normalize timestamp
link_df['normal_time'] = link_df['timestamp'].apply(lambda x: (x - link_df['timestamp'].min()) / \
                                                  (link_df['timestamp'].max() - link_df['timestamp'].min()))

addrs = list(set(list(link_df['sender']) + list(link_df['receiver'])))

addr2id = {addr: idx for idx, addr in enumerate(sorted(addrs))}

addr2id_path = "./dataset/tornado-rule-100/addr2id.json"
with open(addr2id_path, "w") as f:
    json.dump(addr2id, f)


n = len(addrs)
print("Number of nodes:", n)

# use np.digitize to decide which bin
# interval: left closed, right open, final is both inclusive
bins = 100
q = np.linspace(0, 1, num=bins + 1)
link_df['slice'] = np.digitize(link_df['normal_time'], bins=q)
df_split = [link_df[link_df['slice'] == i] for i in range(1, bins + 1)]
# remove empty dataframe
df_split = [df for df in df_split if not df.empty]

label_df = pd.read_csv(f'./dataset/tornado-rule-noedgeattanddiraction/edges.csv')
label_df_split = []

np.random.seed(42)
feature_df = pd.read_csv('./dataset/tornado-rule-noedgeattanddiraction/node_feature_normalized.csv')
node2fea = {}
for row in feature_df.itertuples():
    node2fea[row.node] = np.array(row[2:53])

csv_files = [
    "./dataset/Concat/0.1ETH-concat.csv",
    "./dataset/Concat/1ETH-concat.csv",
    "./dataset/Concat/10ETH-concat.csv",
    "./dataset/Concat/100ETH-concat.csv"
]

cmd = ["python", "input_embedding4.py", addr2id_path] + csv_files
subprocess.run(cmd)

print("节点特征生成完毕")


# Initialize a list to store the number of labels for each time slice
label_counts = []

average_densities = []

for i, split in enumerate(df_split):
    # create graph
    m = len(split)
    print(m)
    edge_index = np.zeros((2, m), dtype=int)

    # random edge features
    # edge_feature = np.random.randn(m, 64)

    #  Value_IN, Value_OUT, GasPrice, normal_time
    value_in = split['Value_IN'].values.reshape(-1, 1)
    value_out = split['Value_OUT'].values.reshape(-1, 1)
    gas_price = split['GasPrice'].values.reshape(-1, 1)
    normal_time = split['normal_time'].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    value_in_normalized = scaler.fit_transform(value_in)
    value_out_normalized = scaler.fit_transform(value_out)
    gas_price_normalized = scaler.fit_transform(gas_price)
    normal_time_normalized = scaler.fit_transform(normal_time)

    edge_feature = np.concatenate([value_in_normalized, value_out_normalized,
                                   gas_price_normalized, normal_time_normalized], axis=1)

    self_loop_feats = np.zeros((len(value_in_normalized), 4))

    edge_set = set()
    edge_time = np.zeros(m, dtype=int)

    self_loop_index = []
    for node_id in range(len(value_in_normalized)):
        self_loop_index.append([node_id, node_id])

    edge_index = np.column_stack([edge_index, np.array(self_loop_index).T])

    # 将自环边的特征（四维零向量）添加到边特征中
    edge_feature = np.vstack([edge_feature, self_loop_feats])

    for j, row in enumerate(split.itertuples()):
        # 可以在这里改一下方向
        edge_index[:, j] = [addr2id[row.sender], addr2id[row.receiver]]

        sender_idx = addr2id[row.sender]
        receiver_idx = addr2id[row.receiver]

        # Add edge (sender -> receiver) if it's not already added
        edge_pair = tuple(sorted([sender_tcidx, receiver_idx]))  # Sort to ensure undirected edge
        # Check if this edge has already been added
        if edge_pair not in edge_set:
            edge_set.add(edge_pair)
        edge_time[j] = row.timestamp
    # Convert edge_index list to numpy array
    edge_index = np.array(list(edge_set)).T

    np.save(f'./dataset/tornado-rule-100/edge_index/{i}.npy', edge_index)
    np.save(f'./dataset/tornado-rule-100/edge_feature/{i}.npy', edge_feature)
    np.save(f'./dataset/tornado-rule-100/edge_time/{i}.npy', edge_time)

    # create label
    min_time = split['timestamp'].min()
    max_time = split['timestamp'].max()
    # handle the last bin separately
    if i < bins - 1:
        label_df_split.append(label_df[(label_df['timestamp'] >= min_time) & (label_df['timestamp'] < max_time)])
    else:
        label_df_split.append(label_df[(label_df['timestamp'] >= min_time) & (label_df['timestamp'] <= max_time)])

    label_edge_index = np.zeros((2, len(label_df_split[i])), dtype=int)
    for j, row in enumerate(label_df_split[i].itertuples()):
        # label_edge_index[:, j] = [addrs.index(row.sender), addrs.index(row.receiver)]
        label_edge_index[:, j] = [addr2id[row.sender], addr2id[row.receiver]]

    label_edge_index = torch.from_numpy(label_edge_index)
    print("Label edges:", label_edge_index.shape)

    label_count = len(label_df_split[i])
    label_counts.append(label_count)

    neg_edge_index = negative_sampling(label_edge_index, num_nodes=n, num_neg_samples=None, method='sparse')
    all_edges = torch.cat([label_edge_index, neg_edge_index], dim=1)
    print("All edges:", all_edges.shape)
    #
    label = np.zeros(all_edges.shape[1], dtype=int)
    label[:label_edge_index.shape[1]] = 1
    np.save(f'./dataset/tornado-rule-100/label/{i}.npy', label)
    np.save(f'./dataset/tornado-rule-100/label_edge_index/{i}.npy', all_edges.numpy())

    # Calculate average density for this time slice
    actual_edges = len(edge_set)
    possible_edges = n * (n - 1) // 2  # For undirected graph
    avg_density = actual_edges / possible_edges
    average_densities.append(avg_density)

# Calculate the average number of labels across all time slices
sum_label_counts = sum(label_counts)
average_label_count = sum(label_counts) / len(label_counts)
print("label_counts:", label_counts)
print("sum_label_counts:", sum_label_counts)
print("Average number of labels per time slice:", average_label_count)

# Print average densities for each time slice
print("Average densities:", average_densities)
print("Overall average density:", sum(average_densities) / len(average_densities))

