
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from dataset import *
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

        # Baseline BERT and Tokenizer for fact encoder and law encoder
        self.fact_tokenizer = AutoTokenizer(self.arglist.model_version)
        self.law_tokenizer = AutoTokenizer(self.arglist.model_version)
        self.fact_model = AutoModel(self.arglist.model_version)
        self.law_model = AutoModel(self.arglist.model_version)

        # Load KLAID Dataset
        self.data = load_dataset("lawcompany/KLAID", 'ljp')
        self.label = pd.read_csv(self.arglist.label_path).iloc[:, :-1]
        self.label.columns = ['laws_service_id', 'laws_service', 'context']

    def parse_default_args(self):
        # BERT related
        self.parser.add_argument("--model_version", default='bert-base-multilingual-cased', type=str)

        # Data related
        self.parser.add_argument("--label_path", default='./klaid_label.csv', type=str)

        # Training related
        self.parser.add_argument("--test_size", default=0.1, type=float)
        self.parser.add_argument("--batch_size", default=16, type=float)

    def parse_args(self):
        self.parse_default_args()

    def train_builder(self):
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
                              data=train)

        valid_dataset = KLAID(fact_tokenizer=self.fact_tokenizer,
                              law_tokenizer=self.law_tokenizer,
                              data=valid)

        # torch DataLoader & Sampler
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   batch_size=self.arglist.batch_size,
                                                   shuffle=False,
                                                   drop_last=True)

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.arglist.batch_size,
                                                   shuffle=False,
                                                   drop_last=True)

        return train_loader, valid_loader
