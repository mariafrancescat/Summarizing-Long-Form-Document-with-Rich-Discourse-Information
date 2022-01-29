import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_size): #(self, embed_size=25*2, embed_size=25*2)
        super().__init__()
        self.embed_size = embed_size
        self.linear = nn.Linear(embed_size*2, embed_size*2)
        self.u_att = nn.Parameter(torch.randn(embed_size*2), requires_grad=True)

    def forward(self, x):
        u = torch.tanh(self.linear(x))
        alpha = F.softmax(torch.matmul(u,self.u_att),dim=1)
        alpha = torch.unsqueeze(alpha,2).repeat(1,1,self.embed_size*2)
        return torch.sum(torch.mul(x,alpha),dim=1)