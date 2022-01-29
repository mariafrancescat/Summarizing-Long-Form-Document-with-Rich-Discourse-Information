from src.datasets.documentDataset import *
from src.models.heroes import *
from src.utils.preprocess import *
from src.utils.dataloader import ArxivDataLoader
from torch.utils.data import DataLoader
from src.losses.contentRankingLoss import ContentRankingLoss

config = {
    'embedding_dim': 25,
    'max_sections': 4,
    'max_sentences_per_section':100,
    'padding': 30
}


data = DocumentDataset('./data/arxiv_train_subset.json', config['padding'])
train_dataloader = ArxivDataLoader(data.documents, batch_size=2, shuffle=True,
    padding=config['padding'], max_sections=config['max_sections'],max_sentences=config['max_sentences_per_section'])

print('Creating model')
model = ContentRanking(data.tokenizer,config['embedding_dim'])
#model.to('cuda:0')

loss_fn = ContentRankingLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for e in range(10):
    epoch_loss = 0
    for batch, (data, sections_gold, sentences_gold) in enumerate(train_dataloader):
        #data = data.to('cuda:0')
        sections_importance, sentences_importance = model(data)
        loss = loss_fn(sections_importance, sentences_importance, sections_gold, sentences_gold)
        epoch_loss+= loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {e}. Loss: {epoch_loss}')
