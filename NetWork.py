import copy
import torch
import torch.nn as nn


# 4. 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x  # 跳跃连接
        out = self.fc(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


# 5. 定义残差网络
class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)  # 输入层
        self.resblock1 = ResidualBlock(100, 100)  # 第一个残差块
        self.resblock2 = ResidualBlock(100, 100)  # 第二个残差块
        self.fc2 = nn.Linear(100, output_dim)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.resblock1(x)  # 第一个残差块
        x = self.resblock2(x)  # 第二个残差块
        x = self.fc2(x)  # 输出层
        return x


class Agent:
    def __init__(self, input_dim, output_dim, train_loder, test_loder, save_num=100, train_epoch=50):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.NetWork = ResNet(8, 2)
        self.loss = nn.CrossEntropyLoss()
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.NetWork.parameters(), lr=self.lr)
        self.train_epoch = train_epoch
        self.train_loader = train_loder
        self.test_loader = test_loder
        self.save_num = save_num
        self.best_model = ResNet(8, 2)
        self.min_loss = 100
        self.train_num = 0

    def train(self):
        for epoch in range(self.train_epoch):
            self.NetWork.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()  # 梯度归零
                outputs = self.NetWork(inputs)  # 前向传播
                loss = self.loss(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新权重
                running_loss += loss.item()
                self.train_num += 1

            if running_loss < self.min_loss:
                self.best_model = copy.deepcopy(self.NetWork)
                self.min_loss = running_loss

            if self.train_num % self.save_num == 0:
                self.save(self.NetWork, f'./model/model_{self.train_num}.pth')
            print(running_loss)
        self.save(self.best_model, f'./model/best_model.pth')

    def test(self):
        self.NetWork.eval()
        correct = 0
        total = 0
        with torch.no_grad():  # 测试时不需要计算梯度
            for inputs, labels in self.test_loader:
                outputs = self.NetWork(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def save(self, model, path):
        torch.save(model.state_dict(), path)

    def load(self, path):
        self.NetWork.load_state_dict(torch.load(path))
