from torch.utils.data import Dataset

class StandardDataset(Dataset):
    def __init__(self, docs, groundtruth=None):
        assert all([self.__checkDocCompliance(doc) for doc in docs]), 'No correct format for StandardDataset'
        self.docs = docs
        self.groundtruth = groundtruth

    def __checkDocCompliance(self,doc):
        isCompliant = False
        keyCompliance = 'doc' in doc.keys() and 'sections' in doc.keys()
        if keyCompliance:
            keyTypes = isinstance(doc['doc'],int) and isinstance(doc['sections'],list)
            if keyTypes:
                sectionsKeys = all(['title' in s.keys() and 'sentences' in s.keys() for s in doc['sections']])
                if sectionsKeys:
                    sectionTypes = all([isinstance(s['title'],str) and isinstance(s['sentences'],list) for s in doc['sections']])
                    if sectionTypes:
                        sentencesStr = all([all([isinstance(sent,str) for sent in s['sentences']]) for s in doc['sections']])
                        if sentencesStr:
                            isCompliant = True
        return isCompliant