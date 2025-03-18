import os
import argparse
from typing import List, Optional, Tuple, Union
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaForCausalLM, LlamaModel
from transformers import  Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding

from modelutils import get_model, set_training, set_inference
from huggingface_hub import login
from datahelper import DataHelper
from trainerutil import create_trainer

from datautil import set_seed


def main(args):
    login("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")  
    # for reproduction
    set_seed(args.seed)

    model, tokenizer = get_model(base_model=args.base_model, model_class=args.model_class, loss_term=args.loss_term)
    model = set_training(model, args)
    
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model)
    
    # model.half()
    # model.to("cuda")
    
    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if "diff_mask" not in name:
            param.requires_grad = False
        
    # for name, param in model.named_parameters():
    #     if "diff_mask" in name:
    #         param.requires_grad = True
    
    dataset_helper = DataHelper(
            tokenizer,
            args.cutoff_len, args.add_eos_token, args.train_on_inputs,
            args.no_instruction, args.prompt_template_name,
            args.verbose,
            no_template_ratio=0.5)
    if args.data_path.endswith(".json"):
        train_data, val_data = dataset_helper.create_dataset_from_json(
                args.data_path)
    else:
        train_data, val_data = dataset_helper.create_dataset(
                args.data_path, args.val_set_size, args.extra_val_dataset,
                args.cache_dataset_dir, args.partial_dir)
    
    # Create trainer
    trainer = create_trainer(
            model, tokenizer, train_data, val_data, 
            args.num_epochs, args.output_dir,
            show_progress=True, load_best_model=False,
            args=args,
            custom_eval_save_step=args.detailed_extra,
            custom_trainer="iter",
            custom_callback=["loss", ])
    
    trainer.train()
    
    
    
    return
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # seed 
    parser.add_argument("--seed", default=50, type=int)

    # Model Type&Path
    parser.add_argument(
        "--base_model",
        type=str,
        default="output_tune/llama-1-7b/ppl_n10/rm_6_blocks_lora_merge_fp16",
        help="base model name",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="dyllm",
        help="chosse in [dyllm, normal]",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")

    parser.add_argument(
        "--data_path", type=str, default="yahma/alpaca-cleaned", help="data path"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="cache_dir for datasets module"
    )
    parser.add_argument(
        "--partial_dir", type=str, default=None, help="Directory to save/load the partial dataset for C4"
    )
    parser.add_argument(
        "--val_set_size", type=int, default=2000, help="validation set size"
    )
    parser.add_argument(
        "--extra_val_dataset",
        type=str,
        default=None,
        help='validation datasets. Split with ","',
    )
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=4, help="micro batch size"
    )
    parser.add_argument("--num_epochs", type=float, default=5, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="alpaca",
        help="The prompt template to use, will default to alpaca.",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        default=False,
        help="Whether to use the instruction template or not.",
    )
    
    
    
    parser.add_argument(
        "--output_dir", type=str, default="./finetune", help="output directory"
    )
    parser.add_argument(
        "--detailed_extra",
        type=int,
        default=None,
        help="Step size used detailed evaluation and saving for extra stage",
    )
    parser.add_argument(
        "--group_by_length",
        default=False,
        action="store_true",
        help="faster, but produces an odd training loss curve",
    )
    
    
    parser.add_argument("--cutoff_len", type=int, default=256, help="cutoff length")
    parser.add_argument("--add_eos_token", default=False, action="store_true")
    parser.add_argument(
        "--train_on_inputs",
        default=False,
        action="store_true",
        help="Train on inputs. If False, masks out inputs in loss",
    )
    
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose on model structure printing",
    )
    
    parser.add_argument('--cache_dataset_dir', type=str, default="./cache_dataset", help='data cache path')
    
    
    # additional loss term
    # mask, distill, ppl
    parser.add_argument('--loss_term', type=str, default="mask", help='loss function for training')
    
    
    args = parser.parse_args()

    main(args)