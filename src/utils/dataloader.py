import numpy as np
import torch
from torch.utils.data import DataLoader

class ArxivDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=2, shuffle=True, padding=30, max_sections=4, max_sentences=10):
        super(ArxivDataLoader,self).__init__(dataset, batch_size=2, shuffle=True, collate_fn = self.custom_collate)
        self.padding = padding
        self.max_sections = max_sections
        self.max_sentences = max_sentences

    def custom_collate(self, batch: list):
        data = {'titles': [], 'sections': []}
        section_importance = torch.zeros(len(batch),self.max_sections)
        sentence_importance = torch.zeros(len(batch),self.max_sections,self.max_sentences)

        for doc_id,doc in enumerate(batch):
            titles = [ np.zeros(self.padding) for _ in range(self.max_sections)]
            sections = [ [np.zeros(self.padding) for _ in range(self.max_sentences)] for _ in range(self.max_sections)]
            for i,sec in enumerate(doc[:self.max_sections]):
                section_importance[doc_id][i] = torch.tensor(sec.trueImportance)
                titles[i] = sec.title.padded_sentence
                sentences = [ np.zeros(self.padding) for _ in range(self.max_sentences) ]
                for j,sent in enumerate(sec.sentences[:self.max_sentences]):
                    sentence_importance[doc_id][i][j]= torch.tensor(sent.trueImportance)
                    sentences[j] = sent.padded_sentence
                sections[i] = sentences

            data['titles'].append(titles)
            data['sections'].append(sections)
        section_importance = torch.unsqueeze(section_importance, 2)
        return data, section_importance, sentence_importance

