
import torch
import pandas as pd


class KLAID(torch.utils.data.Dataset):
    def __init__(self,
                 fact_tokenizer,
                 law_tokenizer,
                 data: pd.DataFrame,
                 shuffle: bool,
                 sample_rate: float):
        data = data.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
        self.data = data
        if shuffle:
            self.data.sample(frac=1).reset_index(drop=True)
        self.data = data
        self.fact_tokenizer = fact_tokenizer
        self.law_tokenizer = law_tokenizer

        self.fact_seq = fact_tokenizer(list(self.data['fact']),
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors='pt')
        self.law_seq = law_tokenizer(list(self.data['context']),
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        result = {
            'f_input_ids': self.fact_seq['input_ids'][item],
            'f_token_type_ids': self.fact_seq['token_type_ids'][item],
            'f_attention_mask': self.fact_seq['attention_mask'][item],
            'l_input_ids': self.law_seq['input_ids'][item],
            'l_token_type_ids': self.law_seq['token_type_ids'][item],
            'l_attention_mask': self.law_seq['attention_mask'][item],
            'laws_service_id': torch.as_tensor(self.data['laws_service_id'][item], dtype=torch.int32)
        }

        return result
