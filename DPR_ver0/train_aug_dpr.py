
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from train import *
from dataset import *
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


class AugDPR(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "Dense Passage Retrieval for Legal Fact Classification"
        )
        self.parse_args()
        self.arglist = self.parser.parse_args()
        self.device = torch.device(f"cuda:{self.arglist.gpu_num}" if torch.cuda.is_available() else 'cpu')

    def parse_default_args(self):
        # BERT related
        self.parser.add_argument("--model_version", default='bert-base-multilingual-cased', type=str)

        # Data related
        self.parser.add_argument("--label_path", default='./klaid_label.csv', type=str)

        # Training related
        self.parser.add_argument("--gpu_num", default=1, type=int)
        self.parser.add_argument("--test_size", default=0.1, type=float)
        self.parser.add_argument("--batch_size", default=2, type=int)
        self.parser.add_argument("--step_size", default=4, type=int)
        self.parser.add_argument("--lr", default=1e-03, type=float)
        self.parser.add_argument("--gamma", default=0.8, type=float)
        self.parser.add_argument("--num_epochs", default=10, type=int)
        self.parser.add_argument("--weight_decay", default=0.1, type=float)
        self.parser.add_argument("--optimizer", default="AdamW", type=str)
        self.parser.add_argument("--adam_eps", default=1e-08, type=float)

    def parse_args(self):
        self.parse_default_args()

    def baseline_builder(self):
        # Baseline BERT and Tokenizer for fact encoder and law encoder
        fact_tokenizer = AutoTokenizer.from_pretrained(self.arglist.model_version)
        law_tokenizer = AutoTokenizer.from_pretrained(self.arglist.model_version)
        fact_model = AutoModel.from_pretrained(self.arglist.model_version)
        law_model = AutoModel.from_pretrained(self.arglist.model_version)

        return fact_tokenizer, law_tokenizer, fact_model, law_model

    def train_builder(self):
        # Load KLAID Dataset
        self.data = load_dataset("lawcompany/KLAID", 'ljp')
        self.label = pd.read_csv(self.arglist.label_path).iloc[:, :-1]
        self.label.columns = ['laws_service_id', 'laws_service', 'context']

        fact = []
        service_id = []
        context = []

        for i in tqdm(range(len(self.data['train']))):
            temp = self.data['train'][i]

            fact.append(temp['fact'])
            service_id.append(temp['laws_service_id'])
            context.append(self.label.iloc[temp['laws_service_id'], 2])

        # Build train dataset with columns ['fact', 'laws_service_id', 'context']
        df = pd.DataFrame({'fact': fact,
                           'laws_service_id': service_id,
                           'context': context})

        # Split dataset into train/test
        xtr, xte, ytr, yte = train_test_split(df.loc[:, ['fact', 'context']],
                                              df.loc[:, 'laws_service_id'],
                                              test_size=self.arglist.test_size,
                                              stratify=df.loc[:, 'laws_service_id'])
        train = pd.concat([xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
        valid = pd.concat([xte.reset_index(drop=True), yte.reset_index(drop=True)], axis=1)

        return train, valid

    def dataloader_builder(self, train, valid):
        # torch Dataset
        train_dataset = KLAID(fact_tokenizer=self.fact_tokenizer,
                              law_tokenizer=self.law_tokenizer,
                              data=train,
                              shuffle=False)
        valid_dataset = KLAID(fact_tokenizer=self.fact_tokenizer,
                              law_tokenizer=self.law_tokenizer,
                              data=valid,
                              shuffle=False)

        # Corpus of 177 Labels
        label_corpus = list(self.label['context'])

        # torch DataLoader & Sampler
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   batch_size=self.arglist.batch_size,
                                                   shuffle=False,
                                                   drop_last=True)

        return train_loader, valid_dataset, label_corpus

    def train_dpr(self):
        print("\n=====     Dense Passage Retrieval for Legal Services     =====")
        print("\n** Active GPU Number : ", self.arglist.gpu_num)

        # 1. Bring Pretrained Models
        print("\n1. Load Pretrained BERT Model for Encoder...")
        self.fact_tokenizer, self.law_tokenizer, self.fact_model, self.law_model = self.baseline_builder()

        # 1. Build Train / Valid Dataset
        print("\n2. Making Datasets...")
        train, valid = self.train_builder()

        # 2. Build DataLoader
        print("\n3. Making DataLoaders...")
        train_loader, valid_dataset, label_corpus = self.dataloader_builder(train=train, valid=valid)

        # 3. Train DPR Model
        print("\n4. Training Dense Passage Retrieval for Legal Services...")
        loss_history, top_1, top_5, top_10, top_25 = train_model(train_dataloader=train_loader,
                                                                 valid_dataset=valid_dataset,
                                                                 label_corpus=label_corpus,
                                                                 law_tokenizer=self.law_tokenizer,
                                                                 fact_model=self.fact_model,
                                                                 law_model=self.law_model,
                                                                 epochs=self.arglist.num_epochs,
                                                                 batch_size=self.arglist.batch_size,
                                                                 scheduler=optim.lr_scheduler.StepLR,
                                                                 weight_decay=self.arglist.weight_decay,
                                                                 adam_eps=self.arglist.adam_eps,
                                                                 step_size=self.arglist.step_size,
                                                                 gamma=self.arglist.gamma,
                                                                 lr=self.arglist.lr,
                                                                 device=self.device)

        # 4. Return Valid Accuracy and Train Loss
        print("\n4. Analyzing Final Result...")
        print(f"    Top  1 Retrieval Accuracy :", top_1[-1])
        print(f"    Top  5 Retrieval Accuracy :", top_5[-1])
        print(f"    Top 10 Retrieval Accuracy :", top_10[-1])
        print(f"    Top 25 Retrieval Accuracy :", top_25[-1])

if __name__ == '__main__':
    dpr = AugDPR()
    dpr.train_dpr()
