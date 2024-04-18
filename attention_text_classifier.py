
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.Q = nn.Linear(in_feat, out_feat)
        self.K = nn.Linear(in_feat, out_feat)
        self.V = nn.Linear(in_feat, out_feat)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        d = K.shape[-1]
        QK_d = (Q @ K.transpose(1, 2)) / (d) ** 0.5
        prob = self.softmax(QK_d)
        attention = prob @ V
        return attention


class AddNorm(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, residual):

        residual = residual[:, :, :x.size(2)]
        return self.layer_norm(x + residual)


class Attention_Classification_Model(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Attention_Classification_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size)
        self.add_norm = AddNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        attention_output = self.attention(x)
        x = self.add_norm(attention_output, x)
        x_attn_out = x.clone()
        x, _ = x.max(dim=1)
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1).repeat(1, attention_output.size(1), 1)
        x = self.add_norm(x,x_attn_out)
        x = F.relu(self.fc2(x))
        x,_ = self.fc3(x).max(dim = 1)
        x = torch.sigmoid(x)
        return x