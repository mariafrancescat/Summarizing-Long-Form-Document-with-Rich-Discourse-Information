from src.datasets.documentDataset import *
from src.models.heroes import *
from src.utils.preprocess import *
from src.utils.dataloader import ArxivDataLoader
from torch.utils.data import DataLoader
from src.losses.contentRankingLoss import ContentRankingLoss
from src.utils.configReader import ConfigReader, ModelConfig
from src.utils.outputManager import OutputManager
import wandb
import sys

sys.setrecursionlimit(10000)
config = ConfigReader()

outputManager = OutputManager(config.output_folder)
if config.wandb:
    wandb.init(
        project="Summarizing-Long-Form-Document-with-Rich-Discourse-Information", 
        entity="riz98",
        config = {})
    wandb.config.update(config.getConfigDict())

if config.logging:
    outputManager.writeLog(config.getConfigDict())


for modelConfig in config.models:
    if modelConfig.do_train:
        dataset = modelConfig.training['dataset']['class'](**modelConfig.training['dataset']['params']) 
        dataloader = modelConfig.training['dataloader']['class'](dataset,**modelConfig.training['dataloader']['params'])
        
        validationset = modelConfig.training['validation']['class'](**modelConfig.training['validation']['params'],
            validation_set=True, pre_trained_tokenizer = dataset.tokenizer)
        validation_loader = modelConfig.training['dataloader']['class'](validationset,**modelConfig.training['dataloader']['params'])
        
        loss = modelConfig.training['loss']['class'](**modelConfig.training['loss']['params'])
        loss = loss.to(config.device)
        if modelConfig.modelParams['tokenizer']:
            model = modelConfig.modelClass(dataset.tokenizer,**modelConfig.modelParams['params'])
        else:
            model = modelConfig.modelClass(**modelConfig.modelParams['params'])
        model = model.to(config.device)
        optimizer = modelConfig.training['optimizer']['class'](model.parameters(),**modelConfig.training['optimizer']['params'])
        #TODO: how to manage wrapper in dataset?
        trainer = modelConfig.training['training_class'](
            model,dataloader, validation_loader, 
            modelConfig.training['epochs'],
            loss, optimizer, config, outputManager
        )
        trainer.train()
    else:
        model = None #TODO: load model from modelConfig.pretrained_model
    if modelConfig.do_inference:
        pass
        #TODO: implementing method

