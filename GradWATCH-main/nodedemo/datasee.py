import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def time_encoding(t, d=16):
    """
    Generate time encoding vector based on sinusoidal functions.
    t: time value (float or tensor)
    d: embedding dimension
    """
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()  # 如果输入是Tensor，转换为NumPy数组

    encoding = []
    for i in range(d):
        if i % 2 == 0:  # Cosine for even dimensions
            encoding.append(np.cos(t / 10000 ** (i / d)))
        else:  # Sine for odd dimensions
            encoding.append(np.sin(t / 10000 ** (i / d)))
    return np.array(encoding).T  # 转置以匹配输入批量数据

# 1. 定义 LearnableFeatureGenerator
class LearnableFeatureGenerator(nn.Module):
    def __init__(self, embedding_dim, position_dim, sigma = 0.1 ):
        super(LearnableFeatureGenerator, self).__init__()
        self.method_embedding = nn.Embedding(2, embedding_dim)  # Deposit 和 Withdraw
        self.value_linear = nn.Linear(1, embedding_dim)  # 交易值映射到嵌入
        self.position_linear = nn.Linear(1, position_dim)  # 位置信息编码
        self.time_linear = nn.Linear(1, embedding_dim)  # 时间编码
        # self.noise = nn.Parameter(torch.randn(embedding_dim * 2 + position_dim + embedding_dim))  # 噪声参数
        self.noise = np.random.normal(0, sigma, size=embedding_dim*2)
        self.embedding_dim = embedding_dim

        # 解码器部分
        self.decoder_method = nn.Linear(embedding_dim * 2 + position_dim + embedding_dim, 2)  # 重构 method
        self.decoder_value = nn.Linear(embedding_dim * 2 + position_dim + embedding_dim, 1)  # 重构 value
        self.decoder_position = nn.Linear(embedding_dim * 2 + position_dim + embedding_dim, 1)  # 重构 position
        self.decoder_time = nn.Linear(embedding_dim * 2 + position_dim + embedding_dim, 1)  # 重构 time

    def forward(self, method_code, value, position, time_seconds):
        method_embed = self.method_embedding(method_code.long())
        value_embed = self.value_linear(value.view(-1, 1))
        position_embed = self.position_linear(position.float().view(-1, 1))
        # time_embed = self.time_linear(time_seconds.view(-1, 1))

        time_embed = torch.tensor(
            time_encoding(time_seconds, d=self.embedding_dim), dtype=torch.float32, device=time_seconds.device
        )

        # 编码器生成嵌入
        # combined = torch.cat([method_embed, value_embed, position_embed, time_embed], dim=-1)
        # combined_with_noise = combined + self.noise

        # print(method_embed.shape)  # 输出形状
        # print(value_embed.shape)  # 输出形状
        # print(torch.cat([method_embed, value_embed], dim=-1).shape)  # 拼接后的形状
        #
        # combined = torch.cat([method_embed, value_embed]) + self.noise
        # combined_with_noise = np.concatenate([combined, time_embed, position_embed.flatten()])

        # 动态将 NumPy 的 noise 转换为 PyTorch 张量
        noise_tensor = torch.tensor(self.noise, dtype=torch.float32, device=method_embed.device)

        # 确保嵌入维度匹配
        combined = torch.cat([method_embed, value_embed], dim=-1)  # 拼接 method 和 value
        combined_with_noise_ = combined + noise_tensor # 加入噪声
        combined_with_noise = torch.cat([combined_with_noise_, position_embed, time_embed], dim=-1)  # 拼接所有特征

        # 解码器重构
        reconstructed_method = self.decoder_method(combined_with_noise)
        reconstructed_value = self.decoder_value(combined_with_noise)
        reconstructed_position = self.decoder_position(combined_with_noise)
        reconstructed_time = self.decoder_time(combined_with_noise)

        return combined_with_noise, reconstructed_method, reconstructed_value, reconstructed_position, reconstructed_time


# 2. 定义重构损失函数
def reconstruction_loss(inputs, reconstructions):
    method_code, value, position, time_seconds = inputs
    reconstructed_method, reconstructed_value, reconstructed_position, reconstructed_time = reconstructions

    # 分类损失（method_code）
    method_loss = nn.CrossEntropyLoss()(reconstructed_method, method_code.long())
    # 回归损失（value, position, time_seconds）
    value_loss = nn.MSELoss()(reconstructed_value, value.view(-1, 1))
    position_loss = nn.MSELoss()(reconstructed_position, position.view(-1, 1))
    time_loss = nn.MSELoss()(reconstructed_time, time_seconds.view(-1, 1))

    # 总损失
    return method_loss + value_loss + position_loss + time_loss


# 3. 加载和预处理数据
df_1 = pd.read_csv(r'D:\2-code\WinGNN\dataset\Concat\0.1ETH-concat.csv')  # 替换为实际文件路径
df_2 = pd.read_csv(r'D:\2-code\WinGNN\dataset\Concat\1ETH-concat.csv')
df_3 = pd.read_csv(r'D:\2-code\WinGNN\dataset\Concat\10ETH-concat.csv')
df_4 = pd.read_csv(r'D:\2-code\WinGNN\dataset\Concat\100ETH-concat.csv')
df = pd.concat([df_1, df_2, df_3, df_4])

# 特征处理
df['method_code'] = df['Method'].apply(lambda x: 1 if x.lower() == 'deposit' else 0)
df['time_seconds'] = (pd.to_datetime(df['DateTime']) - pd.Timestamp("1970-01-01")).dt.total_seconds()
df['position_index'] = df.index

# 归一化处理
df['value_scaled'] = (df['Value_IN'].fillna(0) - df['Value_IN'].mean()) / df['Value_IN'].std()
df['position_scaled'] = (df['position_index'] - df['position_index'].mean()) / df['position_index'].std()
df['time_scaled'] = (df['time_seconds'] - df['time_seconds'].mean()) / df['time_seconds'].std()

# 转换为 Tensor
method_codes = torch.tensor(df['method_code'].values, dtype=torch.float32)
values = torch.tensor(df['value_scaled'].values, dtype=torch.float32)
positions = torch.tensor(df['position_scaled'].values, dtype=torch.float32)
time_seconds = torch.tensor(df['time_scaled'].values, dtype=torch.float32)

# 4. 初始化模型和优化器
embedding_dim = 128
position_dim = 16
feature_generator = LearnableFeatureGenerator(embedding_dim, position_dim)
optimizer = optim.Adam(feature_generator.parameters(), lr=1e-3)


# 5. 训练模型
# num_epochs = 50
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#
#     # 前向传播
#     embeddings, recon_method, recon_value, recon_position, recon_time = feature_generator(
#         method_codes, values, positions, time_seconds
#     )
#
#     # 计算损失
#     inputs = (method_codes, values, positions, time_seconds)
#     reconstructions = (recon_method, recon_value, recon_position, recon_time)
#     loss = reconstruction_loss(inputs, reconstructions)
#
#     # 反向传播与优化
#     loss.backward()
#     optimizer.step()
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
#
# # 6. 验证嵌入
# print(f"最终生成的交易嵌入维度: {embeddings.shape}")
#
# # 7. 聚合节点嵌入
# df['transaction_embedding'] = embeddings.detach().numpy().tolist()
#
# user_embeddings = df.groupby('From')['transaction_embedding'].apply(
#     lambda x: np.mean(np.stack(x), axis=0)
# )
# user_embeddings = user_embeddings.reset_index()
# user_embeddings.columns = ['User', 'User_Embedding']
#
# # 8. 保存节点嵌入
# # 存储到 node_feature
# addrs = list(set(list(df['From']) + list(df['To'])))
# n = len(addrs)
# print("Number of nodes:", n)
#
# node2fea = {}
# feature_df = pd.read_csv('./user_embeddings.csv')
# for row in feature_df.itertuples():
#     # 将多行字符串的嵌入值解析为 NumPy 数组
#     embedding_str = row.User_Embedding.replace("\n", "").strip()  # 去掉换行符和多余空格
#     user_embedding = np.fromstring(embedding_str[1:-1], sep=' ')  # 去掉首尾的括号并解析为数组
#     node2fea[row.User] = user_embedding  # 将结果存入字典
#     # node2fea[row.User] = np.array(row[2:53])
#
# node_dim = 544
#
# node_feature = np.zeros((n, node_dim))   #每个节点特征维数为544维
# for i, addr in enumerate(addrs):
#     if addr not in node2fea:
#         node_feature[i] = np.random.randn(node_dim)
#     else:
#         node_feature[i] = node2fea[addr]
# print(node_feature.shape)
# for i in range(50):
#     np.save(f'./node_feature/{i}.npy', node_feature)
# print("节点特征已保存")



# 5. 训练模型并保存原始与重构数据
num_epochs = 50
original_data = []
reconstructed_data = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 前向传播
    embeddings, recon_method, recon_value, recon_position, recon_time = feature_generator(
        method_codes, values, positions, time_seconds
    )

    # 保存原始与重构数据
    original_data.append({
        'method_code': method_codes.detach().cpu().numpy(),
        'value': values.detach().cpu().numpy(),
        'position': positions.detach().cpu().numpy(),
        'time_seconds': time_seconds.detach().cpu().numpy(),
    })

    reconstructed_data.append({
        'method_code': torch.argmax(recon_method, dim=-1).detach().cpu().numpy(),
        'value': recon_value.detach().cpu().numpy().flatten(),
        'position': recon_position.detach().cpu().numpy().flatten(),
        'time_seconds': recon_time.detach().cpu().numpy().flatten(),
    })

    # 计算损失
    inputs = (method_codes, values, positions, time_seconds)
    reconstructions = (recon_method, recon_value, recon_position, recon_time)
    loss = reconstruction_loss(inputs, reconstructions)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# 9. 可视化数据分布
def plot_distribution(original, reconstructed, title, xlabel):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(original, label='Original', color='blue')
    sns.kdeplot(reconstructed, label='Reconstructed', color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
#
# 获取最后一个 epoch 的原始与重构数据
final_original = original_data[-1]
final_reconstructed = reconstructed_data[-1]
#
# 绘制每个特征的分布对比
plot_distribution(final_original['value'], final_reconstructed['value'], "Value Distribution", "Value")
plot_distribution(final_original['position'], final_reconstructed['position'], "Position Distribution", "Position")
plot_distribution(final_original['time_seconds'], final_reconstructed['time_seconds'], "Time Distribution", "Time Seconds")

# 使用 t-SNE 将数据降维到 3 维
# def reduce_to_3d_tsne(data, perplexity=30, learning_rate=200, n_iter=1000):
#     """
#     使用 t-SNE 将高维数据降到 3 维
#     data: 输入数据 (NumPy 数组)
#     perplexity: t-SNE 的参数，控制局部邻域大小
#     learning_rate: 学习率
#     n_iter: 最大迭代次数
#     """
#     tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
#     reduced_data = tsne.fit_transform(data)
#     return reduced_data
#
#
# # 合并输入数据
# def combine_features(method_code, value, position, time_seconds):
#     return np.stack([method_code, value, position, time_seconds], axis=1)
#
#
# # 原始数据
# original_combined = combine_features(
#     final_original['method_code'],
#     final_original['value'],
#     final_original['position'],
#     final_original['time_seconds']
# )
# # 重构数据
# reconstructed_combined = combine_features(
#     final_reconstructed['method_code'],
#     final_reconstructed['value'],
#     final_reconstructed['position'],
#     final_reconstructed['time_seconds']
# )
#
# # 降维
# original_3d_tsne = reduce_to_3d_tsne(original_combined)
# reconstructed_3d_tsne = reduce_to_3d_tsne(reconstructed_combined)
#
#
# # 3D 可视化
# def plot_3d_tsne(original, reconstructed, title):
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 原始数据分布
#     ax.scatter(original[:, 0], original[:, 1], original[:, 2],
#                c='blue', alpha=0.6, label='Original', s=20)
#     # 重构数据分布
#     ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
#                c='orange', alpha=0.4,  label='Reconstructed', s=20)
#
#     ax.set_title(title)
#     ax.set_xlabel('t-SNE Component 1, fontsize=12')
#     ax.set_ylabel('t-SNE Component 2, fontsize=12')
#     ax.set_zlabel('t-SNE Component 3, fontsize=12')
#     ax.legend()
#     plt.show()
#
#
# # 绘制最终嵌入的分布图
# plot_3d_tsne(original_3d_tsne, reconstructed_3d_tsne, "3D Distribution of Original and Reconstructed Data (t-SNE)")
