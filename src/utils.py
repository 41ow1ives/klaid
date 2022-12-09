import os
import time
import random
import numpy as np
import torch
import re

def set_seed(seed=417):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def clean(x):    
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+')
    x = pattern.sub(' ', x)
    x = x.replace('  ', ' ')
    x = x.replace('( )', '')
    x = x.replace('()', '')
    x = x.strip()
    return x

def save_dpr(fact_model, law_model, epoch, score, arglist):
    # 6. Save Models
    # print("\n6. Save Models")
    fact_dir = os.path.join(arglist.model_dir, 'fact_model')
    law_dir = os.path.join(arglist.model_dir, 'law_model')
    os.makedirs(fact_dir, exist_ok=True)
    os.makedirs(law_dir, exist_ok=True)
    fact_dir = fact_dir + "/" + time.strftime("%m_%d_%H", time.localtime(time.time())) + f'epoch{epoch}_score{score}.pt'
    law_dir = law_dir + "/" + time.strftime("%m_%d_%H", time.localtime(time.time())) + f'epoch{epoch}_score{score}.pt'
    torch.save(fact_model.state_dict(), fact_dir)
    torch.save(law_model.state_dict(), law_dir)    