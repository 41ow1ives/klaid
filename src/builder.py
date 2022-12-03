import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch

from src.dataset import *


class Builder:
    def __init__(self, args):
        self.arglist = args
    
    def baseline_builder(self):
            # Baseline BERT and Tokenizer for fact encoder and law encoder
        self.fact_tokenizer = AutoTokenizer.from_pretrained(self.arglist.model_version)
        self.law_tokenizer = AutoTokenizer.from_pretrained(self.arglist.model_version)
        self.fact_model = AutoModel.from_pretrained(self.arglist.model_version)
        self.law_model = AutoModel.from_pretrained(self.arglist.model_version)

        return self.fact_tokenizer, self.law_tokenizer, self.fact_model, self.law_model

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