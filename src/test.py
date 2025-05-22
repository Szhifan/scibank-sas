from transformers import AutoTokenizer, T5ForConditionalGeneration
from data_prep import SB_Dataset_conditional_generation,LABEL2ID
from utils import eval_report
import json
import pandas as pd
sb = SB_Dataset_conditional_generation()
test_ua = sb.data_dict["test_ua"]
gold_ids = [LABEL2ID["3-ways"].get(g,0) for g in test_ua["label"]]
path_pred = "results/gen/test_ua_generated.txt"
with open(path_pred, "r") as f:
    pred = f.readlines()
    pred = [line.strip() for line in pred] 

pred_ids = [LABEL2ID["3-ways"].get(p,-1) for p in pred]
print(-1 in pred_ids)