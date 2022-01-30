import os
import torch

class OutputManager:
    def __init__(self, path):
        self.__logFile = open(f'{path}/training_log.log','w')
        self.modelPath = f'{path}/model.pt'
    
    def writeLog(self,s):
        if isinstance(s,dict):
            s = [f'{k} -- {v}\n' for k,v in s.items()]
        elif isinstance(s,str):
            s = [f'{s}\n']
        for e in s:
            self.__logFile.write(e)
        self.__logFile.flush()
    
    def saveModel(self,model):
        torch.save(model.state_dict(), self.modelPath)

