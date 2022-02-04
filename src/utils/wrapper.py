from src.models import *

class Wrapper:

    @staticmethod
    def wrapperTo(toClass):
        mapping = {
            Bart : Wrapper.toBart
        }
        return mapping[toClass]

    @staticmethod
    def toBart(standardDataset):
        '''
        receives -> [
            'doc' -> doc_id
            'sections' -> [{'title':str, 'sentences':[str]}]
        ]
        output -> huggingface dataset:2 columns:'Summary', 'Text'
        '''
        from datasets import Dataset

        d = {'Summary':[], 'Text':[]}
        for doc, gt in zip(standardDataset.docs,standardDataset.groundtruth):
            sections = doc['sections']
            for section in sections:
                sentences_concat = ' '.join(section['sentences'])
                d['Summary'].append(sentences_concat)
                d['Text'].append(gt)

        return Dataset.from_dict(d)
