import json
from src.models.heroes import *
from src.utils.dataloader import *
from src.utils.inference import *
from src.trainings import *
from src.losses import *
from src.utils.wrapper import *
import torch
import shutil
import os
import copy

class ModelConfig:
    def __init__(self,model, device):
        self.modelClass, modelConfig = ConfigReader.modelMapper(model)
        self.do_train = model['train']
        self.pretrained_model = model['pretrained_model']
        self.do_inference = model['inference']
        self.tokenizer = ConfigReader.tokenizerMapped(modelConfig['tokenizer'])
        self.modelParams = ConfigReader.modelParamsMapped(modelConfig['params'])
        self.training = {
            'training_class': ConfigReader.trainingMapped(modelConfig['train']['training_class']),
            'dataset': ConfigReader.datasetMapped(modelConfig['train']['training_dataset'],self.modelClass,self.tokenizer),
            'validation': ConfigReader.datasetMapped(modelConfig['train']['validation_dataset'],self.modelClass,self.tokenizer),
            'dataloader': ConfigReader.dataloaderMapped(modelConfig['train']['training_dataloader'],device),
            'epochs': modelConfig['train']['training_epochs'],
            'optimizer': ConfigReader.optimizerMapper(modelConfig['train']['optimizer']),
            'loss': ConfigReader.lossMapped(modelConfig['train']['loss'])
        }
        self.inference = ConfigReader.inferenceMapped(modelConfig['inference'],self.modelClass)
    
    def get_dict(self):
        d = {
            'name': self.modelClass.__name__,
            'pretrained': self.pretrained_model,
            'tokenizer': self.tokenizer['class'].__name__,
            'params': self.modelParams,
            'training_set': self.training['dataset']['params']['data_path'],
            'validation_set': self.training['validation']['params']['data_path'],
            'epochs': self.training['epochs'],
            'optimizer': self.training['optimizer']['class'].__name__,
            'optimizer_params': self.training['optimizer']['params'],
            'loss': self.training['loss']['class'].__name__
        }
        return d

class ConfigReader:
    def __init__(self): 
        with open('./config/config.json','r') as f:
            self.__config = json.load(f)
        f.close()
        self.device = self.__config['device']
        self.models = [ModelConfig(model,self.device) for model in self.__config['models']]
        self.__outputSetup(self.__config['output'])

    def getConfigDict(self):
        total_config = copy.deepcopy(self.__config)
        total_config['models'] = []
        for m in self.models:
            total_config['models'].append(m.get_dict())
        return total_config

    @staticmethod
    def modelMapper(model):
        models = {
            'ContentRanking': ContentRanking,
        }
        m = models[model['name']]
        model_config = json.load(open(f'./config/{model["config"]}','r'))
        return m, model_config

    @staticmethod
    def optimizerMapper(opt):
        optimizers = {
            'SGD': torch.optim.SGD,
        }
        to_return = {
            'params':dict([(k,v) for k,v in opt.items() if k!='name'])
        }
        to_return['class'] = optimizers[opt['name']]
        return to_return

    @staticmethod
    def lossMapped(data):
        losses = {
            'ContentRankingLoss': ContentRankingLoss
        }
        to_return = {
            'params': dict([(k,v) for k,v in data.items() if k!='name'])
        }
        to_return['class'] = losses[data['name']]
        return to_return
    
    @staticmethod
    def trainingMapped(data):
        trainings = {
            'ContentRankingTraining': ContentRankingTraining
        }
        return trainings[data]

    @staticmethod
    def datasetMapped(data, fromClass, tokenizer):
        from src.datasets.documentDataset import DocumentDataset
        datasets = {
            'DocumentDataset': DocumentDataset
        }
        to_return = {
            'params': dict([(k,v) for k,v in data.items() if k!='class' and k!='from_wrapper' and k!='tokenizer'])
        }
        if tokenizer!=None:
            to_return['params']['tokenizer'] = tokenizer

        to_return['class'] = datasets[data['class']]
        if data['from_wrapper']!=None:
            to_return['from_wrapper'] = ConfigReader.wrapperMapped(data['from_wrapper'],fromClass)
        else:
            to_return['from_wrapper'] = None
        return to_return

    @staticmethod
    def wrapperMapped(data,fromClass):
        toClass,_ = ConfigReader.modelMapper(data)
        return Wrapper.wrapperFromTo(fromClass,toClass)

    @staticmethod
    def dataloaderMapped(data, device):
        dataloaders = {
            'ArxivDataLoader': ArxivDataLoader
        }
        to_return = dict([(k,v) for k,v in data.items() if k!='class'])

        to_return['class'] = dataloaders[data['class']]
        to_return['params']['device'] = device
        return to_return
    
    @staticmethod
    def tokenizerMapped(data):
        from src.utils.preprocess import Arxiv_preprocess
        if data==None: return
        tokenizers = {
            'Arxiv_preprocess': Arxiv_preprocess
        }
        to_return = dict([(k,v) for k,v in data.items() if k!='class'])
        to_return['class'] = tokenizers[data['class']]
        return to_return
    
    @staticmethod
    def modelParamsMapped(data):
        to_return = {
            'params':dict([(k,v) for k,v in data.items() if k!='tokenizer'])
        }
        to_return['tokenizer'] = data['tokenizer']
        return to_return


    @staticmethod
    def inferenceMapped(data, fromClass):
        to_return = dict([(k,v) for k,v in data.items() if k!='from_wrapper'])
        to_return['method'] = Inference.inferenceFrom(fromClass)
        if data['from_wrapper']!=None:
            to_return['from_wrapper'] = ConfigReader.wrapperMapped(data['from_wrapper'], fromClass)
        else:
            to_return['from_wrapper'] = None
        return to_return


    def __outputSetup(self,data):
        self.output_folder = f'./outputs/{data["folder"]}'
        if os.path.isdir(self.output_folder):
            print("Overwrite output folder")
            shutil.rmtree(self.output_folder)
        os.mkdir(self.output_folder)
        self.save_model = data['save_model']
        self.logging = data['logging']
        self.wandb = data['wandb']
            