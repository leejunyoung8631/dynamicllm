import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from datasets import load_dataset

import random
import json
import csv



class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)



def get_wikitext2():
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return traindata, testdata


def get_ptb():
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
    return traindata, valdata



def process_data(samples, tokenizer, seq_len, field_name, add_bos_to_every=False):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if not add_bos_to_every:  # add bos token to only the first segment
        test_ids = torch.cat(
            (torch.LongTensor([tokenizer.bos_token_id]), test_ids), dim=0
        )

    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len  
    

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            batch = torch.cat(
                (torch.LongTensor([tokenizer.bos_token_id]), batch), dim=0
            )
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)

    return IndexDataset(tensors=test_ids_batch)





def get_loaders(name, tokenizer, seq_len=2048, batch_size=8, add_bos_to_every=False):
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
    if "ptb" in name:
        train_data, test_data = get_ptb()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "sentence", add_bos_to_every
        )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_data, test_loader





def tokenlayer_table(tokenizer, file="tokeninfo.csv",):
    index_to_word = {idx: word for word, idx in tokenizer.get_vocab().items()}
    
    # read data
    df = pd.read_csv(file, header=None)  # shape = (length, 32)    
    value = np.array(df.values.tolist())
    # value = value.transpose()
    
    detokenized_data = np.vectorize(index_to_word.get)(value)
    
    # for dd in detokenized_data:
        # print(dd)
    # exit()
    
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.axis('off')  # Turn off the axes
    # ax.axis('tight')  # Fit table within the figure

    # Create the table
    table = ax.table(cellText=detokenized_data,  # Round values to 2 decimals
                    loc='center',  # Center the table
                    cellLoc='center')  # Center text within cells

    # Adjust font size and column width
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.auto_set_column_width(col=list(range(value.shape[1])))

    # Show the table
    plt.show()
    plt.tight_layout()
    plt.savefig("./layerwise_prediction.png")
    
    

def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    


def convert_json2csv_zeroshot(json_path, csv_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    select_key = {
        "boolq": "acc",
        "piqa": "acc",
        "hellaswag": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "openbookqa": "acc_norm",
    }

    list_task = []
    list_metric = []
    list_score = []

    ave_score = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for name, key in select_key.items():
            if name not in data["results"]:
                continue

            list_task.append(name)
            list_metric.append(key)

            score = data["results"][name][key] * 100
            list_score.append(score)
            ave_score += score

        ave_score /= len(list_score)

        list_task.append("AVE")
        list_metric.append("n/a")
        list_score.append(ave_score)

        writer.writerow(list_task)
        writer.writerow(list_metric)
        writer.writerow(list_score)

    print(csv_path)