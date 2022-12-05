from src.builder import Builder
import torch
import json
from src.utils import set_seed
from src.args import parse_inference_args
from tqdm import tqdm
import pandas as pd
import argparse

class AugDPR(object):
    def __init__(self, arglist):
        self.arglist = arglist
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.label = pd.read_csv(self.arglist.label_path).iloc[:, :-1]
        self.label.columns = ['laws_service_id', 'laws_service', 'context']
        

    def predict_dpr(self):
        print("\n=====     Dense Passage Retrieval for Legal Services     =====")
        print("\n** Active GPU Number : ", self.arglist.gpu_num)
        
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        builder = Builder(self.arglist)
        # 1. Bring Pretrained Models
        print("\n1. Load Pretrained BERT Model for Encoder...")
        
        fact_tokenizer, law_tokenizer, fact_model, law_model = builder.baseline_builder()
        label_corpus = list(self.label['context'])
        
        # Read Input
        with open('dev_data.json', encoding='UTF-8') as json_file:
                fact = json.load(json_file)
            
        fact_model = fact_model.to('cpu')
        law_model = law_model.to(device=device)
        
        with torch.no_grad():
            
            law_model.eval()
            
            # Create Law Embedding vectors
            law_embs = []
            for law in label_corpus:
                tokenized_label = law_tokenizer(law, padding='max_length', truncation=True, return_tensors='pt').to(device=device)
                embedded_label = law_model(**tokenized_label).pooler_output.cpu()
                law_embs.append(embedded_label.squeeze())
                
            
            fact_model.eval()
            fact_model.to('cpu') # CudaMemoryErr
            
            fact_seq = fact_tokenizer(fact,
                                    padding = 'max_length',
                                    truncation = True,
                                    return_tensors='pt')
            

            print("\n2. Inference...")
            output = []
            for i in tqdm(range(len(fact))):
                embedded_fact = fact_model(fact_seq['input_ids'][i].unsqueeze(0).detach().cpu(),
                                        fact_seq['token_type_ids'][i].unsqueeze(0).detach().cpu(),
                                        fact_seq['attention_mask'][i].unsqueeze(0).detach().cpu()).pooler_output.cpu()
                
                # Calculate Similarity Score with 177 labels
                valid_sim_scores = torch.zeros(len(law_embs))
                for j, emb in enumerate(law_embs):
                    score = torch.matmul(embedded_fact, emb)
                    valid_sim_scores[j] = score
                    
                # Add class
                output += [torch.argmax(valid_sim_scores).item()]
                
                
        # Write json file
        with open('./res.json', 'w') as file:
            json.dump(output, file)
        return 1
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dense Passage Retrieval for Legal Fact Classification")
    arglist = parse_inference_args(parser)
    set_seed(arglist.seed)
    dpr = AugDPR(arglist)
    dpr.predict_dpr()

        