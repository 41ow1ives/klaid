import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW



def train_model(train_dataloader: torch.utils.data.DataLoader,
                fact_model: nn.Module,
                law_model: nn.Module,
                batch_size,
                scheduler: optim.lr_scheduler,
                num_accumulation_step: int,
                weight_decay: float,
                adam_eps: float,
                step_size: int,
                gamma: float,
                lr: float,
                device):

    # Tracking Loss
    train_loss_history = []

    # Model to CUDA
    fact_model = fact_model.to(device=device)
    law_model = law_model.to(device=device)
    

    # Optimizer & Scheduler Setting
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in fact_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in fact_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in law_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in law_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
    scheduler = scheduler(optimizer=optimizer, step_size=step_size, gamma=gamma)

    # Train Log
    start = time.time()
    print("=====          DPR Training Started          =====\n")

    fact_model.train()
    law_model.train()

    for i, dict in tqdm(enumerate(train_dataloader)):
        # Fact Encoder Input
        f_input_ids = dict['f_input_ids'].to(device=device, dtype=torch.long)
        f_token_type_ids = dict['f_token_type_ids'].to(device=device, dtype=torch.long)
        f_attention_mask = dict['f_attention_mask'].to(device=device, dtype=torch.long)

        # Law Encoder Input
        l_input_ids = dict['l_input_ids'].to(device=device, dtype=torch.long)
        l_token_type_ids = dict['l_token_type_ids'].to(device=device, dtype=torch.long)
        l_attention_mask = dict['l_attention_mask'].to(device=device, dtype=torch.long)

        # Model Output([CLS] Vector)
        fact_output = fact_model(f_input_ids, f_token_type_ids, f_attention_mask).pooler_output
        law_output = law_model(l_input_ids, l_token_type_ids, l_attention_mask).pooler_output

        # Similarity Score(Inner Product)
        sim_scores = torch.matmul(fact_output, torch.transpose(law_output, 0, 1))
        # print("before log_softmax :", sim_scores)
        sim_scores = F.log_softmax(sim_scores, dim=1)
        # print(sim_scores)

        # Calculate NLL Loss
        targets = torch.arange(0, batch_size).long()
        targets = targets.to(device=device)

        loss = F.nll_loss(sim_scores, targets)
        loss.backward()

        # Temp Log
        print(f"Batch Number {i:5d} Finished! - Loss = {loss.item():1.5f}")

        # Update
        if ((i + 1) % num_accumulation_step == 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
        elapsed_time = time.time() - start

    # Scheduler
    scheduler.step()

    # Logging
    train_loss_history.append(loss.item())

    return train_loss_history, elapsed_time, fact_model, law_model


def valid_model(valid_dataset: torch.utils.data.Dataset,
                label_corpus: list,
                law_tokenizer,
                fact_model: nn.Module,
                law_model: nn.Module,
                elapsed_time: int,
                epoch: int,
                loss: list,
                device):
    
    top_1_history = []
    top_5_history = []
    top_10_history = []
    top_25_history = []
    
    fact_model = fact_model.to(device=device)
    law_model = law_model.to(device=device)
    
    # Start Validation

    with torch.no_grad():
        # Embed 177 labels
        law_model.eval()
        law_embs = []

        for law in label_corpus:
            tokenized_label = law_tokenizer(law, padding='max_length', truncation=True, return_tensors='pt').to(device=device)
            embedded_label = law_model(**tokenized_label).pooler_output.cpu()
            law_embs.append(embedded_label.squeeze())
            
        # Accuracy of Top N Ranked Labels
        top_1, top_5, top_10, top_25 = 0, 0, 0, 0

        # Embed Facts in Valid Loader
        fact_model.eval()
        fact_model.to('cpu')

        for fact in valid_dataset:
            embedded_fact = fact_model(fact['f_input_ids'].unsqueeze(0).detach().cpu(),
                                       fact['f_token_type_ids'].unsqueeze(0).detach().cpu(),
                                       fact['f_attention_mask'].unsqueeze(0).detach().cpu()).pooler_output.cpu()
            # Calculate Similarity Score with 177 labels
            valid_sim_scores = torch.zeros(len(law_embs))
            for i, emb in enumerate(law_embs):
                score = torch.matmul(embedded_fact, emb)
                valid_sim_scores[i] = score

            # Sorting
            rank = valid_sim_scores.sort()[0]

            # Update Accuracy of Top N Ranked
            if fact['laws_service_id'] == rank[0]:
                top_1 += 1
            if fact['laws_service_id'] in rank[0:5]:
                top_5 += 1
            if fact['laws_service_id'] in rank[0:10]:
                top_10 += 1
            if fact['laws_service_id'] in rank[0:25]:
                top_25 += 1

        top_1_history.append(top_1 / len(valid_dataset))
        top_5_history.append(top_5 / len(valid_dataset))
        top_10_history.append(top_10 / len(valid_dataset))
        top_25_history.append(top_25 / len(valid_dataset))

    loss = np.mean(loss)
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] Epoch {epoch + 1:3d}  Train Loss: {loss:6.5f} | Top 1 Accuracy : {top_1 / len(valid_dataset):1.5f} | Top  5 Accuracy : {top_5 / len(valid_dataset):1.5f} | Top 10 Accuracy : {top_10 / len(valid_dataset):1.5f}")

    return top_1_history, top_5_history, top_10_history, top_25_history