import os
import torch
import time
import wandb
import argparse
import torch.optim as optim
from src.train import *
from src.utils import set_seed
from src.args import parse_default_args
from src.builder import Builder
from src.logger import wandb_init

class AugDPR(object):
    def __init__(self, arglist):
        self.arglist =  arglist
        self.device = torch.device("cuda:" if torch.cuda.is_available() else 'cpu')

    def train_dpr(self):
        print("\n=====     Dense Passage Retrieval for Legal Services     =====")
        print("\n** Active GPU Number : ", self.arglist.gpu_num)

        builder = Builder(self.arglist)
        # 1. Bring Pretrained Models
        print("\n1. Load Pretrained BERT Model for Encoder...")
        fact_tokenizer, law_tokenizer, fact_model, law_model = builder.baseline_builder()

        # 2. Build Train / Valid Dataset
        print("\n2. Making Datasets...")
        train, valid = builder.train_builder()

        # 3. Build DataLoader
        print("\n3. Making DataLoaders...")
        train_loader, valid_dataset, label_corpus = builder.dataloader_builder(train=train, valid=valid)

        # 4. Train DPR Model
        print("\n4. Training Dense Passage Retrieval for Legal Services...")
        
        wandb_init(fact_model, law_model, self.arglist)
        for epoch in range(self.arglist.num_epochs):
            loss_history, elapsed_time = train_model(train_dataloader=train_loader,
                                                     fact_model=fact_model,
                                                     law_model=law_model,
                                                     batch_size=self.arglist.batch_size,
                                                     scheduler=optim.lr_scheduler.StepLR,
                                                     weight_decay=self.arglist.weight_decay,
                                                     adam_eps=self.arglist.adam_eps,
                                                     step_size=self.arglist.step_size,
                                                     gamma=self.arglist.gamma,
                                                     lr=self.arglist.lr,
                                                     device=self.device)
            top_1, top_5, top_10, top_25 = valid_model(valid_dataset=valid_dataset,
                                                       label_corpus=label_corpus,
                                                       law_tokenizer=law_tokenizer,
                                                        fact_model=fact_model,
                                                        law_model=law_model,
                                                        elapsed_time=elapsed_time,
                                                        epochs=epoch,
                                                        device=self.device)
        wandb.finish()


        # 5. Return Valid Accuracy and Train Loss
        print("\n5. Analyzing Final Result...")
        print(f"    Top  1 Retrieval Accuracy :", top_1[-1])
        print(f"    Top  5 Retrieval Accuracy :", top_5[-1])
        print(f"    Top 10 Retrieval Accuracy :", top_10[-1])
        print(f"    Top 25 Retrieval Accuracy :", top_25[-1])
        return fact_model, law_model
        
    def save_dpr(self, fact_model, law_model):
        # 6. Save Models
        print("\n6. Save Models")
        fact_dir = os.pathjoin(self.arglist.model_dir, 'fact_model')
        law_dir = os.pathjoin(self.arglist.model_dir, 'law_model')
        os.makedirs(fact_dir, exist_ok=True)
        os.makedirs(law_dir, exist_ok=True)
        fact_dir = fact_dir + time.strftime('%Y%m%d%H', time.time()) + '.pt'
        law_dir = law_dir + time.strftime('%Y%m%d%H', time.time()) + '.pt'
        torch.save(fact_model.state_dict(), fact_dir)
        torch.save(law_model.state_dict(), law_dir)    
        
        
        

if __name__ == '__main__':
    set_seed(417)
    wandb.init(project="KLAID",
               entity="77601251")
    parser = argparse.ArgumentParser("Dense Passage Retrieval for Legal Fact Classification")
    arglist = parse_default_args(parser)
    dpr = AugDPR(arglist)
    fact_model, law_model = dpr.train_dpr()
    dpr.save_dpr(fact_model, law_model)
