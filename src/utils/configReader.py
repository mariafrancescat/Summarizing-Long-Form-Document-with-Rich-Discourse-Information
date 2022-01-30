import json
from src.models.heroes import *
from src.losses import *
import torch
import shutil
import os
import copy

class ConfigReader:
    def __init__(self):      
        with open('./config/config.json','r') as f:
            self.__config = json.load(f)
        f.close()
        self.device = self.__config['device']
        self.training_epochs = self.__config['training_epochs']
        self.training_batch = self.__config['training_batch']
        self.__modelMapper(self.__config["model"])
        self.__optimizerMapper(self.__config["optimizer"])
        self.__datasetMapped(self.__config["training_dataset"])
        self.__lossMapped(self.__config['loss'])
        self.__OutputSetup(self.__config['output'])

    def getConfigDict(self):
        total_config = copy.deepcopy(self.__config)
        total_config.update({'model_params':self.model_params})
        return total_config

    def __modelMapper(self,model):
        models = {
            'ContentRanking': ContentRanking,
        }
        self.model = models[model['name']]
        self.model_params = json.load(open(f'./config/{model["config"]}','r'))
    
    def __optimizerMapper(self,opt):
        optimizers = {
            'SGD': torch.optim.SGD,
        }
        self.optimizer = optimizers[opt['name']]
        self.optimizer_params = dict([(k,v) for k,v in opt.items() if k!='name'])
 
    def __datasetMapped(self,data):
       self.file_path = f'./data/{data["file_path"]}'
       self.max_sections = data['max_sections']
       self.max_sentences_per_section = data['max_sentences_per_section']
       self.padding = data['padding']

    def __lossMapped(self, data):
        losses = {
            'ContentRankingLoss': ContentRankingLoss
        }
        self.loss = losses[data]
    
    def __OutputSetup(self,data):
        self.output_folder = f'./outputs/{data["folder"]}'
        if os.path.isdir(self.output_folder):
            print("Overwrite output folder")
            shutil.rmtree(self.output_folder)
        os.mkdir(self.output_folder)
        self.save_model = data['save_model']
        self.logging = data['logging']
        self.wandb = data['wandb']
            