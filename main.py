import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from NetWork import ResNet, ResidualBlock, Agent


def data_create(num):
    np.random.seed(42)
    X = np.random.randint(0, 1000, size=(num, 1))  # 生成0到1000之间的随机整数
    y = (X % 2).flatten()  # 奇数为1，偶数为0
    # 2. 转换整数为二进制表示
    X_binary = np.unpackbits(X.astype(np.uint8), axis=1)  # 将整数转换为8位二进制特征
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.3, random_state=42)

    # 4. 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建PyTorch数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader


def train(agent, train_epoch):
    agent.train_epoch = train_epoch
    agent.train()
    return agent

def test(agent):
    success = agent.test()
    print(success)

if __name__ == "__main__":
    train_loader, test_loader = data_create(2000)
    agent = Agent(input_dim=8, output_dim=2, train_loder=train_loader, test_loder=test_loader,
                  save_num=100)
    train(agent, 50)
    test(agent)
