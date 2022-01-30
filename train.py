from src.datasets.documentDataset import *
from src.models.heroes import *
from src.utils.preprocess import *
from src.utils.dataloader import ArxivDataLoader
from torch.utils.data import DataLoader
from src.losses.contentRankingLoss import ContentRankingLoss
from src.utils.configReader import ConfigReader
from src.utils.outputManager import OutputManager
import wandb

config = ConfigReader()
outputManager = OutputManager(config.output_folder)
if config.wandb:
    wandb.init(project="Summarizing-Long-Form-Document-with-Rich-Discourse-Information", entity="riz98")
    wandb.config = config.getConfigDict()

if config.logging:
    outputManager.writeLog(config.getConfigDict())

data = DocumentDataset(config.file_path, config.padding)
train_dataloader = ArxivDataLoader(data.documents, batch_size=config.training_batch, shuffle=True,
    padding=config.padding, max_sections=config.max_sections,max_sentences=config.max_sentences_per_section)

print('Creating model')
model = ContentRanking(data.tokenizer,**config.model_params)
model.to(config.device)

loss_fn = config.loss()
optimizer = config.optimizer(model.parameters(), **config.optimizer_params)

for e in range(config.training_epochs):
    epoch_loss = 0
    for batch, (data, sections_gold, sentences_gold) in enumerate(train_dataloader):
        sections_importance, sentences_importance = model(data)
        loss = loss_fn(sections_importance, sentences_importance, sections_gold, sentences_gold)
        epoch_loss+= loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if config.logging:
        outputManager.writeLog(f'Epoch {e}/{config.training_epochs}. Loss: {epoch_loss}')
    if config.wandb:
        wandb.log({"loss": epoch_loss})
        wandb.watch(model)
    if config.save_model:
        outputManager.saveModel(model)
    print(f'Epoch {e}/{config.training_epochs}. Loss: {epoch_loss}')