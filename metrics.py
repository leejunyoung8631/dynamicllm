import argparse
import csv
import os
import time

import numpy as np
import torch
from dataset import get_loaders
from tqdm import tqdm

from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference

from peft import PeftModel

import matplotlib.pyplot as plt
import json


'''
1. the average skipped blocks per word
2. PPL degradation vs Skipped blocks : correlation PPL degradation vs skipped block
'''




def metric_dictionary(token_dict, label, shift_masks):
    shift_masks = shift_masks.cpu().numpy().astype(np.int64)
    for k, v in zip(label, shift_masks):
        if k in token_dict:
            token_dict[int(k)][0] += 1 # value
            token_dict[int(k)][1] += int(v) # value
        else:
            token_dict[int(k)] = [1, int(v)]
    
    return token_dict


def metric_ppl_skip(ori_loss, loss):    
    ppl_ori_loss = np.exp(ori_loss.cpu().numpy().astype(np.float64))
    ppl_loss = np.exp(loss.cpu().numpy().astype(np.float64))
    
    return ppl_loss - ppl_ori_loss





@torch.no_grad()
def llama_eval(model, test_lodaer, device, n_partial_batch):
    nll = []
    ppl_difference = []
    num_skipped = []
    token_dict = dict()
    
    
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        
        # for dictionary
        check_label = batch[:, :-1].reshape(-1).cpu().numpy().astype(np.int64)
        
        # 1. original loss
        ori_output = model(batch, run_parents=True)
        ori_lm_logits = ori_output.logits
        
        ori_shift_logits = ori_lm_logits[:, :-1, :].contiguous()
        ori_shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        ori_loss = loss_fct(
            ori_shift_logits.reshape(-1, ori_shift_logits.size(-1)), ori_shift_labels.view(-1)
        )
        
        # 2. skipped loss
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        
        # 3. get mask info from skipped model
        mask = model.get_mask()
        shift_masks = mask[:, :-1, :].sum(dim=-1).reshape(-1)
        

        # 4. refine data
        ppl_difference.append(metric_ppl_skip(ori_loss, loss)) # get pairs (degraded ppl vs skipped block)
        num_skipped.append(shift_masks.cpu().numpy())
        
        # optional dictionary
        token_dict = metric_dictionary(token_dict, check_label, shift_masks)
        
    
    ppl_difference = np.concatenate(ppl_difference, axis=-1)    
    num_skipped = np.concatenate(num_skipped, axis=-1)    
    

    # print(torch.cat(nlls, dim=-1).mean())
    # ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl_difference, num_skipped, token_dict



def predictor_metric(model, tokenizer, device):
    dataset = ["wikitext2"]
    max_seq_len = 128
    batch_size = 4
    add_bos_to_every = False
    
    _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
    )
    
    ppl_difference, num_skipped, token_dict = llama_eval(model, test_loader, device=device, n_partial_batch=None)
    
    return ppl_difference, num_skipped, token_dict





@torch.no_grad()
def skip_ppl(model, test_loader, device):
    ppl_data = []

    for batch in tqdm(test_loader):
        batch = batch.to(device)[0].reshape(1, -1) # for the first batch
        # 1. mask_skipping operation for chosen mask
        output = model(batch, )
        chosen_idx = model.get_chosen_idx()[0]
                
        # 2. each skip level
        for skip_id in range(11):
            output = model(batch, choose_mask=skip_id)
            lm_logits = output.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            ppl_data.append(np.exp(loss.cpu().numpy().astype(np.float64)))

        break

    ppl_data = np.array(ppl_data)
    
    return ppl_data, chosen_idx



# it will check the ppl differnece between original and each block skipped 
def check_skiping_ppl(model, tokenizer, device):
    dataset = ["wikitext2"]
    max_seq_len = 128
    batch_size = 4
    add_bos_to_every = False
    
    
    _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
    )
    
    ppl_data = skip_ppl(model, test_loader, device)
    
    return ppl_data




@torch.no_grad()
def two_forward(model, test_loader, device):
    ppa = []
    
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        
        # 1. run parents
        output = model(batch, run_parents=True)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        
        
        # 2. run ski model
        output = model(batch, )
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        skip_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        
        skip_loss_np = skip_loss.detach().cpu().numpy()
        loss_np = loss.detach().cpu().numpy()
        
        ppa.extend(skip_loss_np < loss_np)
    
    
    return ppa


def calculate_ppa(model, tokenizer, device):
    dataset = ["wikitext2"]
    max_seq_len = 128
    batch_size = 4
    add_bos_to_every = False
    
    _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
    )
    
    ppa_score = two_forward(model, test_loader, device)
    
    return ppa_score
    
    

def error_propagation(model, tokenizer, device):
    dataset = ["wikitext2"]
    max_seq_len = 128
    batch_size = 4
    add_bos_to_every = False
    
    _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
    )
    
    ppl_data = calculate_chain(model, test_loader, device)
    
    return ppl_data





@torch.no_grad()
def calculate_chain(model, test_loader, device):
    for batch in tqdm(test_loader):
        batch = batch.to(device)  
        batch = batch[0].reshape(1, -1)
        b, n = batch.shape
        ppl_data = []
        
        for t in range(n):
            p_batch = batch[:, :t+1]
            temp_data = []
            
            # 2. each skip level
            for skip_id in range(11):
                
                # output = model(p_batch, choose_mask=skip_id)
                output = model(p_batch, choose_mask=f"set_last_{skip_id}")
                lm_logits = output.logits

                shift_logits = lm_logits[:, :t+1, :].contiguous()
                shift_labels = batch[:, 1:t+2].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                temp_data.append(np.exp(loss.cpu().numpy().astype(np.float64)))
            ppl_data.append(temp_data)
            
            
            if t==20:
                return ppl_data
        
    
    
    return




def get_maximum(model, tokenizer, device):
    dataset = ["wikitext2"]
    max_seq_len = 128
    batch_size = 4
    add_bos_to_every = False
    
    _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
    )
    
    ppl_data = get_best(model, test_loader, device)
    
    return ppl_data


def make_mask_idx(mask_idx, b, skip_id):
    new_mask_idx = torch.zeros((b, 1), dtype=torch.long)
    new_mask_idx[:, :] = skip_id
    
    if mask_idx is not None:
        new_mask_idx = torch.concat([mask_idx, new_mask_idx], dim=-1)
        
    return new_mask_idx



@torch.no_grad()
def get_best(model, test_loader, device):
    for batch in tqdm(test_loader):
        batch = batch.to(device) # b, n
        b, n = batch.shape
        save_ppl = []
        for i in range(b):
            save_ppl.append([])
            
        mask_idx = None 
        
        # token
        for t in range(n):
            p_batch = batch[:, :t+1]
            # print("batch")
            # print(p_batch)
            temp_data = []
            
            # 2. each skip level
            # for skip_id in range(11):
            for skip_id in range(11):
                temp_mask = make_mask_idx(mask_idx, b, skip_id)
                # print("mask_idx => temp_mask")
                # print(temp_mask)
                output = model(p_batch, choose_mask="greedy", mask_idx=temp_mask)
                lm_logits = output.logits

                shift_logits = lm_logits[:, :t+1, :].contiguous()
                shift_labels = batch[:, 1:t+2].contiguous()
                # print(shift_labels)
                

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                
                # bring specific token loss
                # print(loss)
                if t == 0:
                    loss = loss
                else:
                    indices = t + torch.arange(b) * (t + 1)
                    loss = loss[indices]
                # print(loss)
                # print("\n")
                
                temp_data.append(np.exp(loss.cpu().numpy().astype(np.float32)))
            
            # get the last token
            temp_data = np.stack(temp_data)
            
            # save all the ppl to get ranking loss
            for i in range(b):
                save_ppl[i].append(temp_data[:, i])
            min_index = np.argmin(temp_data, axis=0).reshape(4,1)
            
            if mask_idx == None:
                mask_idx = torch.tensor(min_index, dtype=torch.long)
            else:
                min_inddex = torch.tensor(min_index, dtype=torch.long)
                mask_idx = torch.concat([mask_idx, min_inddex], dim=-1)
            
            if t == 4:
                exit()
            
            
            if t==20:
                return 
        
    
    
    return









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="dyllm",
        help="chosse in [dyllm, ...]",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/llama-7b-hf/ppl")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    
    parser.add_argument("--loss_term", default=None)
    parser.add_argument("--mask_weight", default="./baffo32/decapoda-research-llama-7B-hf")
    
    # callback for checking the average number of blocks skipped
    parser.add_argument("--check_count", action="store_true", help="if True, check the number of skipped blocks")
    parser.add_argument("--is_generation", action="store_true", )
    
    # For peft model
    parser.add_argument("--lora_model", action="store_true", default=False)
    parser.add_argument("--lora_weight", default=None)
    
    parser.add_argument("--savedata", default=None)
    
    args = parser.parse_args()
    
    
    if args.savedata is not None and os.path.exists(os.path.join(args.savedata, "skip_ppl.npy")) and os.path.exists(os.path.join(args.savedata, "token_dictionary.json")):
        print("plot using saved data")
        
        # 1. ppl_diff vs skip
        data = np.load(os.path.join(args.savedata, "skip_ppl.npy"))
        lg = len(data) // 2
        n_skip = data[:lg]
        diff_ppl = data[lg:]   
        
        plt.figure(figsize=(15,10))
        plt.scatter(n_skip, diff_ppl)
        plt.ylim(0, 200000)
        plt.savefig(os.path.join(args.savedata, "ppl_skip.png"))
        plt.close()
        
        
        # 2. token dictionary
        values =  []
        with open(os.path.join(args.savedata, "token_dictionary.json"), "r") as f:
            token_dict = json.load(f) 
        x = np.array(list(token_dict.keys()))
        y = np.array(list(token_dict.values()))
        for i, data in enumerate(y):
            values.append(data[1] / data[0])
        
        values = np.array(values)
        
        # sort by y values with decreasing if needed 
        ord = np.argsort(-values)
        values = values[ord]
        x = x[ord]
        
        ind = 100
        x = x[:ind]
        values = values[:ind]
        
        plt.figure(figsize=(20,10))
        plt.bar(x, values, )
        plt.xlabel('Token index')
        plt.xlabel('Skipped number')
        plt.tight_layout()
        plt.savefig(os.path.join(args.savedata, "token_dictionary.png"))
        plt.close()
        
        
        exit()
    
    
    
    from huggingface_hub import login
    login("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")  
    set_seed(args.seed)
    
    
    # init default model
    model, tokenizer = get_model(base_model=args.base_model, model_class=args.model_class, loss_term=args.loss_term, cache_dir="lm_cache")    
    model = load_mask_weight(model, args.mask_weight)
    model = set_inference(model, args)
    model.half()
    model = model.cuda()
    
    
    # init peft model
    if args.lora_model:
        if args.lora_weight == None:
            print("Lora model requires lora model weight")
            print("Give weight path to args.lora_weight")
        model = PeftModel.from_pretrained(model, args.lora_weight)
    
    
    '''
    1. ppl differnece
    '''
    
    # ppl_difference, num_skipped, token_dict = predictor_metric(model, tokenizer, args.device)
    
    
    # save_data = np.concatenate([num_skipped, ppl_difference], axis=1)
    # np.save(os.path.join(args.savedata, "skip_ppl.npy"), save_data)
    
    # with open(os.path.join(args.savedata, "token_dictionary.json"), "w") as f:
    #     json.dump(token_dict, f)
    
    
    '''
    2. ppl diff with all skipped block
    '''
    # cutline = 20
    # ppl_data, chosen_data = check_skiping_ppl(model, tokenizer, args.device)
    # ppl_data = ppl_data[:, :cutline]
    # chosen_data = chosen_data[:cutline]   
    
    # fig, ax = plt.subplots(figsize=(16, 6))  # Adjust size as needed

    # # Hide axes
    # ax.axis('off')
    # ax.axis('tight')
    
    # table = ax.table(cellText=np.round(ppl_data, 2), loc='center', cellLoc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(6)  
    # table.scale(1.2, 1.2)
    
    # highlight_color = "#fff2cc"  # 연한 노란색
    
    # for row_idx, col_idx in enumerate(chosen_data):
    #     cell = table[col_idx, row_idx]  
    #     cell.set_facecolor(highlight_color)
    #     cell.get_text().set_weight('bold')
    
    
    # num_cols = ppl_data.shape[1]
    # for col_idx in range(num_cols):
    #     table[0, col_idx].get_text().set_weight('bold')


    # plt.tight_layout()
    # plt.savefig("./test.png")
    
    
    '''
    3. calculate ppl metric 
    '''
    # ppa = calculate_ppa(model, tokenizer, args.device)
    # ppa_score = np.mean(ppa) * 100
    # print("total token length : ", len(ppa))
    # print("PPA score : ", ppa_score)
    
    
    
    '''
    4. Error propagation
    '''
    # ppl_data = error_propagation(model, tokenizer, args.device)
    # for idx, data in enumerate(ppl_data, start=1):
    #     data = np.array(data)
    #     fig, ax = plt.subplots(figsize=(data.shape[1]*1.5, 6))  # 너비를 열 수에 따라 조정
    #     ax.axis('off')
    #     ax.axis('tight')
        
    #     table = ax.table(cellText=np.round(data, 2), loc='center', cellLoc='center')
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(8)
    #     table.scale(1.2, 1.2)
        
    #     plt.tight_layout()
    #     plt.savefig(f".dirdir/table_{idx}.png")  # 각각 table_1.png, table_2.png, ..., table_20.png 저장
    #     plt.close(fig)  # 메모리 절약을 위해 그림 닫기
    
    
    
    '''
    5. get minimum value answer
    '''
    # get_maximum(model, tokenizer, args.device)
    
    
    
    
    '''
    6. save data with ranking loss information
    '''
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    @torch.no_grad
    def get_dataset(tokenizer, model, device="cpu", max_length=256):
        def extract_token_features(batch):
            # batch["text"] : List[str]
            enc = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            
            print(input_ids)
            print(input_ids.shape)
            exit()            
            

            out = model(input_ids, attention_mask=attn_mask)
            hs = out.last_hidden_state.cpu()  # [B, L, H]

            features = [seq.numpy().tolist() for seq in hs]
            return {"token_features": features}

        return extract_token_features
    
    
    
    extract_fn = get_dataset()
    ds = ds.map(extract_fn, batched=True, batch_size=8, remove_columns=["text"])
    
    
    
    
        
    
    
    
    


    
    
    
    
    
    