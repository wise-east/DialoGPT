import sys 

import torch
import logging
import re
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import time
from prettytable import PrettyTable

def fix_weight_keys(weights): 
    keys = list(weights.keys())
    for k in keys: 
        if 'module.' in k: 
            weights[re.sub('module.', '', k)] = weights[k]
            weights.pop(k, None)
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)
    return weights 

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Load and properly initialize models...")

torch.set_grad_enabled(False)

tokenizer = GPT2Tokenizer('models/medium/vocab.json', 'models/medium/merges.txt')

weights = torch.load(sys.argv[1])
cfg = GPT2Config.from_json_file('models/medium/config.json')
model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
weights = fix_weight_keys(weights)
model.load_state_dict(weights)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)