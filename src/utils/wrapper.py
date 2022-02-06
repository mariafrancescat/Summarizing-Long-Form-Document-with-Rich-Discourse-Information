from src.models import *
from src.datasets import *

class Wrapper:

    @staticmethod
    def wrapperTo(toClass):
        mapping = {
            Bart : Wrapper.toBart
        }
        return mapping[toClass]

    @staticmethod
    def toBart(standardDataset, datasetParams):
        from datasets import Dataset

        d = {'Summary':[], 'Text':[]}
        for doc, gt in zip(standardDataset.docs,standardDataset.groundtruth):
            sections = doc['sections']
            for section in sections:
                sentences_concat = ' '.join(section['sentences'])
                d['Summary'].append(sentences_concat)
                d['Text'].append(gt)

        return BartDataset(**datasetParams, fromWrapper=True, datasetFromWrapper=Dataset.from_dict(d))
