from src.datasets.documentDataset import *
from src.models.heroes import *
from src.utils.preprocess import *
from src.utils.dataloader import ArxivDataLoader
from torch.utils.data import DataLoader
from src.losses.contentRankingLoss import ContentRankingLoss
from src.utils.configReader import ConfigReader, ModelConfig
from src.utils.outputManager import OutputManager
from src.utils.wrapper import Wrapper
import wandb
import sys
import copy

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


for modelIndex in range(len(config.models)):
    modelConfig = config.models[modelIndex]

    if modelConfig.from_previous:
        assert modelIndex>0, 'First model cannot have wrapper!'
        standard = config.models[modelIndex-1].inference['method'](model,dataloader,config.device,**config.models[modelIndex-1].inference['params'])
        dataset = Wrapper.wrapperTo(modelConfig.modelClass)(standard)

        standard = config.models[modelIndex-1].inference['method'](model,validation_loader,config.device,**config.models[modelIndex-1].inference['params'])
        validationset = Wrapper.wrapperTo(modelConfig.modelClass)(standard)
    else:
        dataset = modelConfig.training['dataset']['class'](**modelConfig.training['dataset']['params']) 
        dataloader = modelConfig.training['dataloader']['class'](dataset,**modelConfig.training['dataloader']['params'])

        validationset = modelConfig.training['validation']['class'](**modelConfig.training['validation']['params'],
            validation_set=True, pre_trained_tokenizer = dataset.tokenizer)
        validation_loader = modelConfig.training['dataloader']['class'](validationset,**modelConfig.training['dataloader']['params'])

    if modelConfig.modelParams['tokenizer']:
        model = modelConfig.modelClass(tokenizer=dataset.tokenizer,**modelConfig.modelParams['params'], device=config.device)
    else:
        model = modelConfig.modelClass(**modelConfig.modelParams['params'], device=config.device)
    
    if modelConfig.do_train:
        #TODO: manage bart training       
        loss = modelConfig.training['loss']['class'](**modelConfig.training['loss']['params'],device=config.device)
        optimizer = modelConfig.training['optimizer']['class'](model.parameters(),**modelConfig.training['optimizer']['params'])
        trainer = modelConfig.training['training_class'](
            model,dataloader, validation_loader, 
            modelConfig.training['epochs'],
            loss, optimizer, config, outputManager
        )
        trainer.train()
    else:
        if modelConfig.modelParams['tokenizer']: #There must be an embedding layer -> extract dimension
            embedding_dim = torch.load(modelConfig.pretrained_model)['embedding.embedding.weight'].shape
            model = modelConfig.modelClass(tokenizer = embedding_dim, **modelConfig.modelParams['params'])
        else:
            model = modelConfig.modelClass(**modelConfig.modelParams['params'])
        model.load_state_dict(torch.load(modelConfig.pretrained_model))
    if modelConfig.do_inference:
        inferenceset = modelConfig.inference['dataset']['class'](**modelConfig.inference['dataset']['params'],
        validation_set=True, pre_trained_tokenizer = dataset.tokenizer) 
        inference_loader = modelConfig.inference['dataloader']['class'](inferenceset,**modelConfig.inference['dataloader']['params'])
        docs_summary = modelConfig.inference['method'](model,inference_loader,config.device,**modelConfig.inference['params'])
   
        '''
        TODO
        - Every inference outputs a StandardDataset
        - Every wrapper transforms a StandardDataset into a specific one
        - if model i has wrapper:
            standard = model[i-1](dataset) -> StandardDataset
            dataset = wrapper(model_class)(standard) -> Specific dataset for model i
          else:
            dataset = **read dataset from config** -> Specific dataset for model i
        - Maybe rename DocumentDataset to ContentRankingDataset
        - Add to every document the self.groundtruth containing the raw abstracts
        - if not labelled documents, for StandardDataset self.groundtruth=None
        '''