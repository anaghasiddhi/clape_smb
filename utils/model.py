import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.embedding_layer = nn.Linear(320, 128)

    def forward(self, x):
        # x: [B, L, 1280]
        score = self.projection(x)             # [B, L, 2]
        embedding = self.embedding_layer(x)    # [B, L, 128]
        return score, embedding

class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1280, 1280, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1280)

        self.conv2 = nn.Conv1d(1280, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)

        self.head = nn.Softmax(-1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        inter = x.permute(0, 2, 1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x), inter

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layer1 = nn.GRU(1280, 1024, 1, batch_first=True)
        self.layer2 = nn.GRU(1024, 128, 1, batch_first=True)
        self.layer3 = nn.GRU(128, 64, 1, batch_first=True)

        self.project = nn.Linear(64, 2)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        bz = x.size(0)
        device = x.device
        h0 = torch.zeros(1, bz, 1024).to(device)
        out, _ = self.layer1(x, h0)

        h1 = torch.zeros(1, bz, 128).to(device)
        out, _ = self.layer2(out, h1)

        h2 = torch.zeros(1, bz, 64).to(device)
        out, _ = self.layer3(out, h2)

        score = self.project(out)
        return self.head(score), out

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.begin = nn.Linear(320, 1024)
        self.attention = nn.MultiheadAttention(1024, 8, 0.3, batch_first=True)
        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.3)

        self.ffn = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1024)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.begin(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(self.norm(x + x))
        inter = x
        x = self.dropout(self.norm(self.ffn(x) + x))
        return self.class_head(x), inter
