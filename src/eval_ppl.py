"""
Code modified from
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/evaluator/ppl.py
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import load_dataset
from torch.utils.data.dataset import Dataset


from utils import get_model, set_seed




class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)
    


class IndexTrainerDataset(Dataset):
    def __init__(self, input_ids: torch.LongTensor, pad_token_id: int):
        self.input_ids = input_ids
        self.attention_mask = (input_ids != pad_token_id).long()
        self.labels = input_ids.clone()

    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_mask[idx],
            "labels":          self.labels[idx],
        }






def get_wikitext2():
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="cache_dir")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir="cache_dir")
    return traindata, testdata



def get_wikitext103():
    traindata = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir="cache_dir")
    testdata = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", cache_dir="cache_dir")
    return traindata, testdata


def get_ptb():
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
    return traindata, valdata





def process_data(samples, tokenizer, seq_len, field_name, add_bos_to_every=False):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]),
        return_tensors="pt",
        truncation=True,
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





def process_data_trainer(samples, tokenizer, seq_len, field_name, add_bos_to_every=False):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]),
        return_tensors="pt",
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0]
    if not add_bos_to_every:  # add bos token to only the first segment
        test_ids = torch.cat(
            (torch.LongTensor([tokenizer.bos_token_id]), test_ids), dim=0
        )

    batches = []
    nsamples = test_ids.numel() // seq_len  
    

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            batch = torch.cat(
                (torch.LongTensor([tokenizer.bos_token_id]), batch), dim=0
            )
        batches.append(batch)
        
    # dataset
    input_ids  = torch.stack(batches, dim=0)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    labels = input_ids.clone()
    
    import datasets
    ds = datasets.Dataset.from_dict({
        "input_ids":      input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "labels":         labels.cpu().numpy(),
    })

    return datasets.DatasetDict({"train": ds})




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
    if "wikitext103" in name:
        train_data, test_data = get_wikitext103()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
        

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_data, test_loader







def get_loaders_trainer(name, tokenizer, seq_len=2048, batch_size=8, add_bos_to_every=False):
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2()
        train_dataset = process_data_trainer(
            train_data, tokenizer, seq_len, "text", add_bos_to_every
        )
        test_dataset = process_data_trainer(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
    if "ptb" in name:
        train_data, test_data = get_ptb()
        train_dataset = process_data_trainer(
            train_data, tokenizer, seq_len, "sentence", add_bos_to_every
        )
        test_dataset = process_data_trainer(
            test_data, tokenizer, seq_len, "sentence", add_bos_to_every
        )
    if "wikitext103" in name:
        train_data, test_data = get_wikitext103()
        train_dataset = process_data_trainer(
            train_data, tokenizer, seq_len, "text", add_bos_to_every
        )
        test_dataset = process_data_trainer(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
        

    return train_dataset, test_dataset








@torch.no_grad()
def llama_eval(model, test_lodaer, device, n_partial_batch):
    nlls = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)

        if n_partial_batch is not None and len(nlls) == n_partial_batch:
            break

    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def eval_ppl(
    model,
    tokenizer,
    datasets=["wikitext2", "ptb"],
    max_seq_len=128,
    batch_size=4,
    device="cuda",
    add_bos_to_every=False,
    n_partial_batch=None
):
    filename = "ppl_bos.csv" if add_bos_to_every else "ppl.csv"
    csv_header = []
    csv_value = []
    metric = {}
    for dataset in datasets:
        t0 = time.perf_counter()
        _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
        )
        metric[dataset] = llama_eval(model, test_loader, device, n_partial_batch)

        print(
            f"PPL-{dataset}: {metric[dataset]} | add_bos_to_every: {add_bos_to_every} | time: {time.perf_counter()-t0:.1f}"
        )
        csv_header.append(f"ppl_{dataset}")
        csv_value.append(metric[dataset])

    return csv_value



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--use_8bit_model", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    
    parser.add_argument("--device", type=str, default="cuda", help="gpu settings")
    parser.add_argument("--max_seq_len", type=int, default=128, help="length for dataste")
    parser.add_argument("--batchsize", type=int, default=8, help="batchsize ")
    parser.add_argument("--seed", type=int, default=1234, help="seed value")
    
    args = parser.parse_args()
    
    
    from huggingface_hub import login
    login("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")  
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat, args.use_8bit_model)

    for add_bos_to_every in [False]:
        eval_ppl(
            model=model,
            tokenizer=tokenizer,
            datasets=["wikitext2", ],
            batch_size=args.batchsize,
            max_seq_len=args.max_seq_len,
            device=device,
            add_bos_to_every=add_bos_to_every,
        )
    
