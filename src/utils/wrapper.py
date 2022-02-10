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

        d = {'Summary':[], 'Text':[], 'Doc_id':[]}
        for doc, gt in zip(standardDataset.docs,standardDataset.groundtruth):
            sections = doc['sections']
            for section in sections:
                sentences_concat = ' '.join(section['sentences'])
                d['Summary'].append(gt)
                d['Text'].append(sentences_concat)
                d['Doc_id'].append(doc['doc'])

        return BartDataset(**datasetParams, fromWrapper=True, datasetFromWrapper=Dataset.from_dict(d))
