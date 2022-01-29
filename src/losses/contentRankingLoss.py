import torch
from torch import nn

class ContentRankingLoss(nn.Module):
    def __init__(self):
        super(ContentRankingLoss, self).__init__()
        self.section_loss = nn.BCELoss()
        self.sentence_loss = nn.BCELoss()

    def forward(self, sections_importance, sentences_importance, sections_gold, sentences_gold):
        loss1 = self.section_loss(sections_importance, sections_gold)
        loss2 = self.sentence_loss(sentences_importance, sentences_gold)
        return torch.add(loss1,loss2)