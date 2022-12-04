import os
import torch
import time
import wandb
import argparse
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from src.train import *
from src.utils import set_seed, save_dpr
from src.args import parse_default_args
from src.builder import Builder
from src.logger import wandb_init

class AugDPR(object):
    def __init__(self, arglist):
        self.arglist = arglist
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
        
        best_top_1 = -1
        wandb_init(fact_model, law_model, self.arglist)
        
        for epoch in range(self.arglist.num_epochs):
            loss_history, elapsed_time, fact_model, law_model = train_model(train_dataloader=train_loader,
                                                                            fact_model=fact_model,
                                                                            law_model=law_model,
                                                                            batch_size=self.arglist.batch_size,
                                                                            scheduler=optim.lr_scheduler.StepLR,
                                                                            weight_decay=self.arglist.weight_decay,
                                                                            adam_eps=self.arglist.adam_eps,
                                                                            step_size=self.arglist.step_size,
                                                                            gamma=self.arglist.gamma,
                                                                            lr=self.arglist.lr,
                                                                            num_accumulation_step=self.arglist.num_accumulation_step,
                                                                            device=self.device)
            top_1, top_5, top_10, top_25 = valid_model(valid_dataset=valid_dataset,
                                                       label_corpus=label_corpus,
                                                       law_tokenizer=law_tokenizer,
                                                        fact_model=fact_model,
                                                        law_model=law_model,
                                                        elapsed_time=elapsed_time,
                                                        epoch=epoch,
                                                        loss=loss_history,
                                                        device=self.device)
            wandb.log({'train_loss':np.mean(loss_history),
                       'top_1_mean':np.mean(top_1),
                       'top_5_mean':np.mean(top_5),
                       'top_10_mean':np.mean(top_10),
                       'top_25_mean':np.mean(top_25)})
            
            if best_top_1 <= np.mean(top_1):
                best_top_1 = np.mean(top_1)
                print(f"\n Saving Models... epoch: {epoch}, score: {best_top_1}")
                save_dpr(fact_model, law_model, epoch, best_top_1, arglist)
                
        wandb.finish()


        # 5. Return Valid Accuracy and Train Loss
        print("\n5. Analyzing Final Result...")
        print(f"    Top  1 Retrieval Accuracy :", top_1[-1])
        print(f"    Top  5 Retrieval Accuracy :", top_5[-1])
        print(f"    Top 10 Retrieval Accuracy :", top_10[-1])
        print(f"    Top 25 Retrieval Accuracy :", top_25[-1])
        return fact_model, law_model
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dense Passage Retrieval for Legal Fact Classification")
    arglist = parse_default_args(parser)
    set_seed(arglist.seed)
    wandb.init(project="KLAID-base",
               entity="klaid")
    dpr = AugDPR(arglist)
    fact_model, law_model = dpr.train_dpr()
