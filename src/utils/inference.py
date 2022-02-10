import numpy as np
from src.models import *

class Inference:

    @staticmethod
    def inferenceFrom(modelClass):
        mapping = {
            ContentRanking : Inference.contentRanking,
            Bart : Inference.bart
        }
        return mapping[modelClass]

    @staticmethod
    def contentRanking(model, inference_loader, device, number_of_sections, number_of_sentences):
        from src.datasets import StandardDataset
        assert isinstance(model,ContentRanking), 'Model not instance of ContentRanking'
        raw_docs = inference_loader.dataset.documents
        summary_store = []
        for doc_index, (data, sections_gold, sentences_gold) in enumerate(inference_loader):
            sections_importance, sentences_importance = model(data, device=device)
            sections_importance = sections_importance[0]
            sentences_importance = sentences_importance[0]
            summary_doc = {'doc':doc_index,'sections':[]}
            importance_list = [section.item() for section in sections_importance]
            true_section_number = len(raw_docs[doc_index].sections)
            upper_bound = min(true_section_number,len(importance_list))
            n_sections = min(upper_bound,true_section_number)
            best_setcions = np.argpartition(importance_list[:upper_bound],-n_sections)[-n_sections:]        
            for section_index in best_setcions:
                title = raw_docs[doc_index].sections[section_index].title.sentence
                importance_list = [sent.item() for sent in sentences_importance[section_index]]
                true_sentence_number = len(raw_docs[doc_index].sections[section_index].sentences)
                upper_bound = min(true_sentence_number,len(importance_list))
                n_sentences = min(upper_bound,number_of_sentences)
                best_sentences = np.argpartition(importance_list[:upper_bound],-n_sentences)[-n_sentences:]
                raw_sentences = [raw_docs[doc_index].sections[section_index].sentences[sent_index].sentence for sent_index in best_sentences]
                summary_doc['sections'].append({'title':title, 'sentences':raw_sentences})
            summary_store.append(summary_doc)
        return StandardDataset(summary_store, inference_loader.dataset.groundtruth)

    @staticmethod
    def bart(model, inference_loader, device, number_of_sections, number_of_sentences):
        from src.datasets import StandardDataset
        assert isinstance(model,Bart), 'Model not instance of Bart'
        input_ids_per_doc_storage = dict()
        tokenizer = inference_loader.tokenizer.tokenizer
        for d in inference_loader.dataset:
            doc_id = d['Doc_id'].item()
            input_ids = d['input_ids']
            if doc_id not in input_ids_per_doc_storage.keys():
                input_ids_per_doc_storage[doc_id] = []
            input_ids_per_doc_storage[doc_id]+=input_ids

        summary_store = []
        for doc_id, input_ids in input_ids_per_doc_storage.items():    
            summary_ids = model.generate(torch.unsqueeze(torch.tensor(input_ids),0), num_beams=4, max_length=25, early_stopping=True)[0]
            original = ' '.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in input_ids])
            summary = ' '.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]) 
            summary_store.append({'doc':doc_id,'sections':[{'title':'Summary','sentences':summary.split('.')}]})
        
        return summary_store