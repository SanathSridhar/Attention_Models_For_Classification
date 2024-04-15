
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.Q = nn.Linear(in_feat, out_feat)  # Query
        self.K = nn.Linear(in_feat, out_feat)  # Key
        self.V = nn.Linear(in_feat, out_feat)  # Value
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        d = K.shape[-1]  # dimension of key vector
        QK_d = (Q @ K.transpose(1, 2)) / (d) ** 0.5  # Transpose K
        prob = self.softmax(QK_d)
        attention = prob @ V
        return attention

class Attention_Classification_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Attention_Classification_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x).mean(dim=1)  # Mean pooling over time steps
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return torch.sigmoid(x)
