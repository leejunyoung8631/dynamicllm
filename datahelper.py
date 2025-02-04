# Data Helper, CELL, 2024
# - Copied from retrain_lora and refactored for better readability
import os
import torch
import json
import random
import lm_eval

from collections.abc import Iterator
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value
from LLMPruner.utils.prompter import Prompter

class DataHelper:
    ZeroShotTasks = [
            "openbookqa","arc_easy","winogrande",
            "hellaswag","arc_challenge","piqa","boolq","sciq"]
    
    KorDataset = [
        "dbdu/ShareGPT-74k-ko", 
        "heegyu/korquad-chat-v1", 
        "Bingsu/ko_alpaca_data",
        "beomi/KoAlpaca-v1.1a",
        "heegyu/OIG-small-chip2-ko"
    ]

    def __init__(
            self, tokenizer,
            cutoff_len=256, add_eos_token=False, train_on_inputs=False,
            no_instruction=False, prompt_template_name="alpaca",  # Prompt
            verbose=False, max_output_len=128,
            no_template_ratio = 0.0
            ):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.add_eos_token = add_eos_token
        self.train_on_inputs = train_on_inputs
        self.no_instruction = no_instruction
        self.verbose = verbose
        self.max_output_len = max_output_len
        self.no_template_ratio = no_template_ratio
        
        # Setup tokenizer
        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:  # e.g., llama-3 (https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36#662315ec5d73c1b9f90482ea)
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Setup Prompter (Will not be used for text-only dataset)
        self.prompter = Prompter("alpaca")
        self.zeroshot_prompter = ZeroShotPrompter()

    def create_dataset_from_json(
            self, json_path,
            val_set_size=2000, extra_val_datasets=None):
        # Load the data from the JSON file
        with open(json_path, 'r') as json_file:
            dataset_list = json.load(json_file)

        val_data = None
        train_data_list = []
        for dataset_item in dataset_list:
            data_path = dataset_item["data_path"]
            name = dataset_item["name"]
            size = dataset_item["size"]
            cache_dir = dataset_item["cache_dir"]
            partial_dir = dataset_item["partial_dir"]
            full_load = dataset_item.get("full_load", False)
            if "cfg_name" in dataset_item.keys():
                cfg_name = dataset_item["cfg_name"]
            else:
                cfg_name = None
            print("Loading dataset: " + data_path)

            if data_path == "yahma/alpaca-cleaned":
                # Load validataion data together
                train_data, val_data = self.create_dataset(
                    "yahma/alpaca-cleaned", val_set_size,
                    extra_val_datasets=extra_val_datasets)
                if size is not None:
                    train_data = train_data.select(range(size))
            elif data_path in DataHelper.KorDataset: # as it will not use alpaca-cleaned as validation dataset
                train_data, val_data = self.create_dataset(
                    data_path, val_set_size=val_set_size,
                    train_name=name,
                    train_size=size,
                    cache_dir=cache_dir,
                    partial_dir=partial_dir,
                    cfg_name=cfg_name,
                    full_load=full_load)
            else:
                train_data, _ = self.create_dataset(
                    data_path, val_set_size=0,
                    train_name=name,
                    train_size=size,
                    cache_dir=cache_dir,
                    partial_dir=partial_dir,
                    cfg_name=cfg_name,
                    full_load=full_load)

            train_data_list.append(train_data)

        if val_data is None and val_set_size > 0:
            # Load alpaca
            _, val_data = self.create_dataset(
                    "yahma/alpaca-cleaned", val_set_size,
                    extra_val_datasets=extra_val_datasets)

        # Mix training datasets
        if self.verbose:
            total_len = 0
            for dataset_item, trn_data in zip(dataset_list, train_data_list):
                pretty_name = dataset_item["data_path"]
                if dataset_item["name"] is not None:
                    pretty_name += "({})".format(dataset_item["name"])
                data_len = len(trn_data)
                total_len += data_len
                print("[Dataset Mix] {}: {}".format(pretty_name, data_len))
            print("[Dataset Mix] Total: {}".format(total_len))

        merged_dataset = concatenate_datasets(train_data_list)
        train_data = merged_dataset.shuffle(seed=41)
        return train_data, val_data

    def create_dataset(
            self, data_path,
            val_set_size=2000, extra_val_datasets=None, cache_dir=None, partial_dir=None,
            train_name=None, train_size=None, cfg_name=None, full_load=False):
        val_data = {}
        if data_path == "c4":
            assert partial_dir is not None, "Please give me partial_dir. Using all data samples from c4 takes too much time."

            if os.path.exists(partial_dir):
                print(f"Loading partial dataset from {partial_dir}")
                data = load_dataset("json", data_files={
                    "train": os.path.join(partial_dir, "train.json"),
                    "validation": os.path.join(partial_dir, "validation.json")})
                train_val = {"train": data["train"], "test": data["validation"]}
                if train_size is not None:
                    train_val["train"] = train_val["train"].select(range(train_size))
            else:
                print("Trying to load full C4 dataset. This may take a long time...")
                data = load_dataset("c4", "en", cache_dir=cache_dir)

                # Take a subset of the dataset
                train_data = data["train"].shuffle(seed=42).select(range(200000))
                val_data = data["validation"].shuffle(seed=42).select(range(val_set_size))

                # Save the subset for later use
                os.makedirs(partial_dir, exist_ok=True)
                train_data.to_json(os.path.join(partial_dir, "train.json"))
                val_data.to_json(os.path.join(partial_dir, "validation.json"))

                if train_size is not None:
                    train_data = train_data.select(range(train_size))
                train_val = {"train": train_data, "test": val_data}
        elif data_path == "yahma/alpaca-cleaned":
            # The original logic placed here for reproducibility
            data = load_dataset(data_path, cache_dir=cache_dir)
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            if train_size is not None:
                train_val["train"] = train_val["train"].select(range(train_size))
        elif data_path in DataHelper.ZeroShotTasks:
            # For zeroshot, we use lm_eval as it is easiler to create the prompt
            train_data, _ = self.create_zeroshot_dataset(data_path, train_size)
            train_val = {"train": train_data}
        elif data_path in DataHelper.KorDataset: # korean dataset
            data = self.get_kordataset(data_path)
            train_val = data["train"].train_test_split(
                test_size=val_set_size, # i manually set this
                shuffle=True, seed=42
            )
            if train_size is not None:
                train_val["train"] = train_val["train"].select(range(train_size))
                
        elif data_path == "wikitext":
            from datasets import DatasetDict
            train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
            test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
            train_val = DatasetDict({
                "train": train_data,
                "test": test_data
            })
            
        else:
            # Other datasets
            if train_name is None:
                split = "train"
            else:
                split = train_name
            if not full_load and train_size is not None:
                split += '[:{}]'.format(train_size)
            if cfg_name is None:
                train_data = load_dataset(data_path, split=split, cache_dir=cache_dir)
            else:
                train_data = load_dataset(data_path, cfg_name, split=split, cache_dir=cache_dir)
            if full_load and train_size is not None:
                train_data = train_data.select(range(train_size))
            train_val = {"train": train_data}
            
        # Junyoung
        # divide english and korean dataset preprocessor
        # i just want to remain professor's code almost unchanged :) 
        _generate_and_tokenize_prompt = None
        if data_path in DataHelper.KorDataset:
            from functools import partial
            _generate_and_tokenize_prompt = partial(self._generate_and_tokenize_prompt_kor, data_path=data_path)
        else:
            _generate_and_tokenize_prompt = self._generate_and_tokenize_prompt
            
        train_data = train_val["train"].shuffle().map(_generate_and_tokenize_prompt)
        if "test" in train_val.keys():
            val_data = {
                data_path: train_val["test"].shuffle().map(_generate_and_tokenize_prompt),
            }
            

        if extra_val_datasets:
            extra_val_data = self._create_val_data_from_names(extra_val_datasets)
            val_data.update(extra_val_data)

        return train_data, val_data
    
    def _create_val_data_from_names(self, val_datasets):
        val_data = {}
        seq_len = 128
        for extra_dataset in val_dataset.split(","):
            if "wikitext2" in extra_dataset:
                test_data = load_dataset(
                        "wikitext", "wikitext-2-raw-v1", split="test")
                test_data = self._split_and_tokenizer(
                    test_data, seq_len, field_name="text"
                )
            elif "ptb" in extra_dataset:
                test_data = load_dataset(
                        "ptb_text_only", "penn_treebank", split="validation")
                test_data = self._split_and_tokenizer(
                    test_data, seq_len, field_name="sentence"
                )
            else:
                raise NotImplementedError

            val_data[extra_dataset] = test_data
        return val_data

    def create_zeroshot_dataset(self, task_name_, train_size=None, load_val=False):
        task_names = lm_eval.utils.pattern_match(
                [task_name_], lm_eval.tasks.ALL_TASKS)
        task_dict = lm_eval.tasks.get_task_dict(task_names)
        task_name = list(task_dict.keys())[0]
        task = task_dict[task_name]
        assert task.has_training_docs() and task.has_validation_docs()

        rnd = random.Random()
        rnd.seed(42)

        def _make_data_list(task_docs):
            task_docs = list(task_docs)
            rnd.shuffle(task_docs)
            
            data_points = []
            template_printed = False
            non_template_printed = False
            for doc_id, doc in enumerate(task_docs):
                use_non_template = random.random() < self.no_template_ratio
                if use_non_template:
                    inst_text, input, output_text = _zs_text_without_template(
                            task_name_, task, doc, rnd)
                else:
                    inst_text, input, output_text = _zs_text_for_template(
                            task_name_, task, doc, rnd)

                dp = {
                    "instruction": inst_text,
                    "input": input,
                    "output": output_text,
                    "task": task_name_,
                }

                if self.verbose:  # For Debug
                    if use_non_template and not non_template_printed:
                        non_template_printed = True
                        print("Non-template example: ", dp)
                    elif not use_non_template and not template_printed:
                        template_printed = True
                        print("Template example: ", dp)
                data_points.append(dp)

            dataset = Dataset.from_list(data_points)
            features = Features({
                "instruction": Value(dtype="string"),
                "input": Value(dtype="string"),
                "output": Value(dtype="string"),
                "task": Value(dtype="string"),
            })
            return dataset.cast(features)

        train_data = _make_data_list(task.training_docs())
        if load_val:
            val_data = _make_data_list(task.validation_docs())
        else:
            val_data = None
        if train_size is not None:
            train_data = train_data.select(range(train_size))

        return train_data, val_data
    
    
    
    # Junyoung : 
    # separate the korean dataset load function 
    # because sometimes only a part of the data is neeeded
    def get_kordataset(self, data_path):
        if data_path == "dbdu/ShareGPT-74k-ko": # as it will automatically load dirty & cleaned version
            data = load_dataset(
                data_path,
                data_files={
                    "train": ["part1_ko_cleaned.json", "part2_ko_cleaned.json"]
                }
            )
        else:
            data = load_dataset(data_path,)

        return data
    
    

    def _tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
    

    def _tokenize_prompt_with_output(self, prompt, output):
        # Tokenize output first to get the output size
        tokenized_output = self.tokenizer(
            output,
            truncation=True,
            max_length=self.max_output_len,
            padding=False,
            return_tensors=None,
        )
        output_size = len(tokenized_output["input_ids"])

        # Tokenize input and instruction, truncating to leave space for the output
        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len - output_size,
            padding=False,
            return_tensors=None,
        )

        # Combine prompt and output
        tokenized_full_prompt = {
            "input_ids": tokenized_prompt["input_ids"] + tokenized_output["input_ids"],
            "attention_mask": tokenized_prompt["attention_mask"] + tokenized_output["attention_mask"]
        }

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_output["input_ids"].copy()
        return tokenized_full_prompt
    

    def _generate_and_tokenize_prompt(self, data_point):
        non_instruction_field = None
        if "text" in data_point.keys():  # C4, openwebtext, ...
            non_instruction_field = "text"
        elif "query" in data_point.keys():  # wikitext-103-filtered
            non_instruction_field = "query"
        elif "sentence" in data_point.keys():
            non_instruction_field = "sentence"          
        elif "conversations" in data_point.keys():
            non_instruction_field = "conversations"
                                  
        if non_instruction_field is not None:
            full_prompt = data_point[non_instruction_field]
            tokenized_full_prompt = self._tokenize(full_prompt)
        else:
            if "task" in data_point.keys():
                prompter = self.zeroshot_prompter
                prompter.set_task(data_point["task"])
            else:
                prompter = self.prompter

            input_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"]
            )

            if data_point["output"] != "":
                tokenized_full_prompt = self._tokenize_prompt_with_output(
                        input_prompt, data_point["output"])
            else:
                # For the task has no output, we treat it as a non-instruction sample
                tokenized_full_prompt = self._tokenize(input_prompt)
        return tokenized_full_prompt
    
    
    # dataset has its own structures... :<
    def _generate_and_tokenize_prompt_kor(self, data_point, data_path):
        '''
        (Multi-turn) dbdu/ShareGPT-74k-ko : id | conversations(human, gpt) - 55k
        (Multi-turn) heegyu/korquad-chat-v1 : soruce | text(sys, usr, bot) -  9.6K
        (Single-turn) Bingsu/ko_alpaca_data : instruction | input | output - 	52K
        (Single-turn) beomi/KoAlpaca-v1.1a : intruction(user Question) | output -  21K
        (Single-turn) heegyu/OIG-small-chip2-ko : user_translated | chip2_translated - 	210K
        
        '''
        SYSTEM_SENTENCE = "<|start_header_id|>system<|end_header_id|>" # meta-info
        USER_SENTENCE = "<|start_header_id|>user<|end_header_id|>" # user, input
        ASSISTANT_SENTENCE = "<|start_header_id|>assistant<|end_header_id|>" # gpt, output
        
        tokenized_full_prompt = ""
        if data_path == "dbdu/ShareGPT-74k-ko":
            for data in data_point["conversations"]:
                if data["from"] == 'human':
                    tokenized_full_prompt += USER_SENTENCE
                elif data["from"] == 'gpt':
                    tokenized_full_prompt += ASSISTANT_SENTENCE
                tokenized_full_prompt += data["value"]
                tokenized_full_prompt += "<|eot_id|>"
        elif data_path == "heegyu/korquad-chat-v1":
            data = data_point["text"]
            data = data.split("\n")
            for sen in data:
                role = sen[:5]
                if role == "<sys>":
                    tokenized_full_prompt += SYSTEM_SENTENCE
                elif role == "<usr>":
                    tokenized_full_prompt += USER_SENTENCE
                elif role == "<bot>":
                    tokenized_full_prompt += ASSISTANT_SENTENCE
                tokenized_full_prompt += sen[5:].strip()
                tokenized_full_prompt += " <|eot_id|>"
        elif data_path == "Bingsu/ko_alpaca_data": # added ? -> python code tokenize?
            tokenized_full_prompt += SYSTEM_SENTENCE
            tokenized_full_prompt += data_point["instruction"]
            if data_point["input"] != "":
                tokenized_full_prompt += USER_SENTENCE
                tokenized_full_prompt += data_point["input"]
            tokenized_full_prompt += ASSISTANT_SENTENCE
            tokenized_full_prompt += data_point["output"]
        elif data_path == "beomi/KoAlpaca-v1.1a":
            tokenized_full_prompt += USER_SENTENCE
            tokenized_full_prompt += data_point["instruction"]
            tokenized_full_prompt += ASSISTANT_SENTENCE
            tokenized_full_prompt += data_point["output"]
        elif data_path == "heegyu/OIG-small-chip2-ko":
            tokenized_full_prompt += USER_SENTENCE
            tokenized_full_prompt += data_point["user_translated"]
            tokenized_full_prompt += ASSISTANT_SENTENCE
            tokenized_full_prompt += data_point["chip2_translated"]
        
        
        # possible added component (Optional)
        # save memory space with remove_columns
        # e.g. dataset.remove_columns(['id', 'conversations'])
            
        
        tokenized_full_prompt = self._tokenize(tokenized_full_prompt)        
        return tokenized_full_prompt
    
    

    def _split_and_tokenizer(self, test_data, seq_len, field_name):
        test_ids = self.tokenizer(
            "\n\n".join(test_data[field_name]), return_tensors="pt"
        ).input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
            test_set.append({"input_ids": batch, "labels": batch})
        return test_set


class DataIterator(Iterator):
    def __init__(self,
            tokenizer,
            data : List[Dict[str, List[int]]],  # == <class 'datasets.arrow_dataset.Dataset'>
            batch_size: int,
            device: torch.device,
            n_samples_to_load: int = None,
            add_labels: bool = True,
            add_tasks: bool = False
            ) -> Dict[str, torch.Tensor]:
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.add_labels = add_labels
        self.add_tasks = add_tasks

        self.current_idx = 0
        self.n_samples = len(data)

        if n_samples_to_load is not None:
            self.n_samples = min(n_samples_to_load, self.n_samples)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        self.current_idx += self.batch_size
        minibatch = self.data.select(range(start_idx, end_idx))
        return self._prepare_batch(minibatch)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def _prepare_batch(self, batch):
        """
        Prepares an inference batch by extracting input fields and padding them.
        """
        # Extract relevant fields from batch with padding
        relevant_inputs = [
            {
                'input_ids': data_point['input_ids'],
                'attention_mask': data_point['attention_mask'],
            } for data_point in batch
        ]
        inputs = self.tokenizer.pad(
                relevant_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(self.device)

        # Add labels if required
        if self.add_labels:
            labels = [data_point['labels'] for data_point in batch]
            max_length = inputs['input_ids'].shape[1]
            padded_labels = [
                label + [-100] * (max_length - len(label)) for label in labels
            ]
            inputs['labels'] = torch.tensor(
                    padded_labels, dtype=torch.long, device=self.device)

        if self.add_tasks:
            tasks = []
            for data_point in batch:
                if 'task' in data_point:
                    tasks.append(data_point['task'])
                else:
                    tasks.append(task)
            inputs['task'] = tasks

        return inputs


def _zeroshot_text(input_choices, answer):
    assert isinstance(answer, int)
    assert len(input_choices) > 1
    choice_lst = [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(input_choices)]
    choices_str = "\n".join(choice_lst)
    return choices_str, choice_lst[answer]


def _zs_text_for_template(task_name_, task, doc, rnd):
    if isinstance(task, lm_eval.tasks.winogrande.Winogrande):
        inst_text = doc["sentence"]
        input_choices = [doc["option1"], doc["option2"]]
        answer = int(doc["answer"]) - 1
        input, output_text = _zeroshot_text(input_choices, answer)
    elif isinstance(task, lm_eval.base.MultipleChoiceTask):
        ctx = task.fewshot_context(
            doc=doc, num_fewshot=0, rnd=rnd, description=None
        )
        reqs = task.construct_requests(doc, ctx)

        if not isinstance(reqs, (list, tuple)):
            reqs = [reqs]

        inst_text = None
        input_choices = []
        for req in reqs:
            assert inst_text is None or inst_text == req.args[0]
            inst_text = req.args[0]
            input_choices.append(req.args[1])

        answer = int(doc["gold"])
        input, output_text = _zeroshot_text(input_choices, answer)
    else:
        assert task_name_ == "boolq"
        inst_text = doc["passage"]
        input = doc["question"]
        output_text = task.doc_to_target(doc)
        if output_text.lower().strip() == "yes":
            output_text = "yes"
        else:
            output_text = "no"

    # Remove the unnecessary text (will be included in the template)
    answer_postfix = "\nAnswer:"
    if inst_text.endswith(answer_postfix):
        inst_text = inst_text[:-len(answer_postfix)]

    return inst_text, input, output_text


def _zs_text_without_template(task_name_, task, doc, rnd):
    # Make a short answer question or plain text without using the template
    if isinstance(task, lm_eval.tasks.winogrande.Winogrande):
        # Create only the correct statement so that it does not use template
        sentence = doc["sentence"]
        input_choices = [doc["option1"], doc["option2"]]
        answer = int(doc["answer"]) - 1

        pronoun_loc = sentence.index("_")
        inst_text = sentence[:pronoun_loc] + input_choices[answer] + sentence[pronoun_loc+1:]
        output_text = ""  # We don't give any output for winogrande, should be treated later as a plain text
    elif isinstance(task, lm_eval.base.MultipleChoiceTask):
        ctx = task.fewshot_context(
            doc=doc, num_fewshot=0, rnd=rnd, description=None
        )
        reqs = task.construct_requests(doc, ctx)

        if not isinstance(reqs, (list, tuple)):
            reqs = [reqs]

        inst_text = None
        input_choices = []
        for req in reqs:
            assert inst_text is None or inst_text == req.args[0]
            inst_text = req.args[0]
            input_choices.append(req.args[1])
        answer = int(doc["gold"])

        if not "Question:" in inst_text:
            inst_text = "Question: " + inst_text
        if not inst_text.endswith("\nAnswer:"):
            inst_text += "\nAnswer: "
        output_text = input_choices[answer]
    else:
        inst_text = "{passage}\nQuestion: {question}\nAnswer (yes or no): ".format(
                    passage=doc["passage"], question=doc["question"] 
                )
        output_text = task.doc_to_target(doc)
        if output_text.lower().strip() == "yes":
            output_text = "yes"
        else:
            output_text = "no"

    return inst_text, "", output_text  # Return input with empty string - the ZeroShotPrompter will not use its template.


class ZeroShotPrompter:
    def set_task(self, task):
        self.task = task

    def generate_prompt(self, instruction, input, label=None):
        if input != "":
            templates = ZeroShotPrompter.Templates[self.task]["prompt_variations"]
            prompt_variation = random.choice(templates)
            prompt = prompt_variation.format(instruction=instruction, input=input)
        else:
            # Correct statement
            prompt = instruction + ' '

        if label is not None:
            prompt += label
        return prompt

    Templates = {
            "openbookqa": {
                "description": "Template for OpenBookQA.",
                "prompt_variations": [
                    "Below is a question and several answer choices. Select the correct answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "You are given a question along with multiple choices. Please select the most appropriate answer.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Correct Answer:\n",
                    "Choose the right answer from the following choices based on the question below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Selected Answer:\n",
                    "Select the answer that best explains the question below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "Read the following question and choose the correct answer.\n\n### Question:\n{instruction}\n\n### Possible Answers:\n{input}\n\n### Correct Answer:\n",
                    "Consider the question below and pick the best answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "From the given choices, choose the most appropriate answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Selected Answer:\n",
                    "Below is a multiple-choice question. Please select the answer that you think is correct.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Correct Answer:\n",
                    "Given the question and choices below, select the correct answer.\n\n### Question:\n{instruction}\n\n### Answer Options:\n{input}\n\n### Answer:\n",
                    "Answer the following question by selecting the right choice.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n"
                    ],
                "response_split": "### Answer:"
                },
            "arc_easy": {
                "description": "Template for ARC Easy.",
                "prompt_variations": [
                    "Below is a science question and several answer choices. Select the correct answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "Here is a simple science question along with multiple choices. Choose the correct answer.\n\n### Science Question:\n{instruction}\n\n### Options:\n{input}\n\n### Answer:\n",
                    "Select the correct answer from the following options based on the science question below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Correct Answer:\n",
                    "Read the question below and choose the correct answer from the given choices.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Selected Answer:\n",
                    "Consider the following science question and choose the best answer.\n\n### Science Question:\n{instruction}\n\n### Possible Answers:\n{input}\n\n### Answer:\n",
                    "Choose the answer that best completes the question below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Correct Answer:\n",
                    "Answer the science question by selecting the correct choice.\n\n### Science Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "Select the best answer for the following science question.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Correct Answer:\n",
                    "Given the question below, pick the correct answer from the choices provided.\n\n### Question:\n{instruction}\n\n### Answer Choices:\n{input}\n\n### Answer:\n",
                    "From the options below, select the answer that best fits the science question.\n\n### Science Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n"
                    ],
                "response_split": "### Answer:"
                },
            "arc_challenge": {
                "description": "Template for ARC Challenge.",
                "prompt_variations": [
                    "Below is a challenging science question and several answer choices. Select the correct answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "You are given a challenging science question with multiple choices. Select the most appropriate answer.\n\n### Challenging Science Question:\n{instruction}\n\n### Options:\n{input}\n\n### Correct Answer:\n",
                    "Please choose the correct answer for the following challenging science question.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Selected Answer:\n",
                    "Read the challenging question below and select the best answer.\n\n### Science Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                    "Consider the challenging science problem and pick the correct answer.\n\n### Question:\n{instruction}\n\n### Possible Answers:\n{input}\n\n### Correct Answer:\n",
                    "Select the answer that is most suitable for the question below.\n\n### Science Question:\n{instruction}\n\n### Options:\n{input}\n\n### Answer:\n",
                    "From the given options, choose the correct answer to the challenging science question.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Selected Answer:\n",
                    "Choose the most appropriate answer for the challenging science question provided.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Answer:\n",
                    "Given the challenging question, pick the right answer from the options.\n\n### Science Question:\n{instruction}\n\n### Answer Choices:\n{input}\n\n### Correct Answer:\n",
                    "Answer the challenging science question by selecting the correct answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n"
                    ],
                "response_split": "### Answer:"
                },
            "winogrande": {
                "description": "Template for Winogrande.",
                "prompt_variations": [
                    "Below is a sentence with a missing word. Choose the correct word to complete the sentence.\n\n### Sentence:\n{instruction}\n\n### Answer:\n",
                    "Complete the following sentence by choosing the correct word from the options given.\n\n### Sentence:\n{instruction}\n\n### Correct Word:\n",
                    "Fill in the blank in the sentence below by selecting the correct word.\n\n### Incomplete Sentence:\n{instruction}\n\n### Selected Word:\n",
                    "Choose the word that correctly completes the sentence.\n\n### Sentence:\n{instruction}\n\n### Answer:\n",
                    "Given the sentence below, pick the correct word to fill in the blank.\n\n### Sentence:\n{instruction}\n\n### Correct Word:\n",
                    "Select the correct word to complete the following sentence.\n\n### Sentence:\n{instruction}\n\n### Answer:\n",
                    "From the options, choose the word that best fits the blank in the sentence.\n\n### Sentence:\n{instruction}\n\n### Correct Answer:\n",
                    "Complete the sentence with the most appropriate word.\n\n### Incomplete Sentence:\n{instruction}\n\n### Selected Word:\n",
                    "Read the sentence and choose the correct word to fill in the missing part.\n\n### Sentence:\n{instruction}\n\n### Answer:\n",
                    "Fill in the blank in the sentence below by choosing the most suitable word.\n\n### Sentence:\n{instruction}\n\n### Correct Word:\n"
                    ],
                "response_split": "### Answer:"
                },
            "hellaswag": {
                    "description": "Template for HellaSwag.",
                    "prompt_variations": [
                        "Below is a situation with possible endings. Choose the ending that makes the most sense.\n\n### Situation:\n{instruction}\n\n### Ending:\n{input}\n\n### Answer:\n",
                        "Read the given situation and choose the ending that best fits the context.\n\n### Situation:\n{instruction}\n\n### Possible Endings:\n{input}\n\n### Best Ending:\n",
                        "Choose the most appropriate ending for the following situation.\n\n### Scenario:\n{instruction}\n\n### Endings:\n{input}\n\n### Selected Ending:\n",
                        "Given the situation, select the best possible ending.\n\n### Situation:\n{instruction}\n\n### Endings:\n{input}\n\n### Answer:\n",
                        "Consider the scenario below and pick the ending that fits best.\n\n### Scenario:\n{instruction}\n\n### Possible Endings:\n{input}\n\n### Best Choice:\n",
                        "Which ending makes the most sense for the given situation?\n\n### Situation:\n{instruction}\n\n### Options:\n{input}\n\n### Selected Ending:\n",
                        "Select the ending that logically follows the given scenario.\n\n### Situation:\n{instruction}\n\n### Possible Endings:\n{input}\n\n### Answer:\n",
                        "Choose the appropriate ending based on the context of the situation.\n\n### Situation:\n{instruction}\n\n### Endings:\n{input}\n\n### Best Ending:\n",
                        "Pick the ending that best completes the following situation.\n\n### Scenario:\n{instruction}\n\n### Ending Options:\n{input}\n\n### Answer:\n",
                        "Read the situation and choose the ending that fits most appropriately.\n\n### Situation:\n{instruction}\n\n### Possible Endings:\n{input}\n\n### Selected Ending:\n"
                        ],
                    "response_split": "### Answer:"
                    },
            "piqa": {
                    "description": "Template for PIQA.",
                    "prompt_variations": [
                        "Below is a physical reasoning question. Choose the answer that best fits the given situation.\n\n### Question:\n{instruction}\n\n### Answer:\n",
                        "Select the best answer for the given physical reasoning question.\n\n### Physical Question:\n{instruction}\n\n### Correct Answer:\n",
                        "Choose the most reasonable answer based on the following physical reasoning problem.\n\n### Problem:\n{instruction}\n\n### Selected Answer:\n",
                        "Given the physical reasoning question, pick the best answer.\n\n### Question:\n{instruction}\n\n### Answer:\n",
                        "Read the problem and choose the answer that makes the most sense.\n\n### Physical Question:\n{instruction}\n\n### Correct Answer:\n",
                        "Consider the physical scenario and choose the best answer from the options.\n\n### Scenario:\n{instruction}\n\n### Selected Answer:\n",
                        "Select the most appropriate answer for the physical reasoning scenario below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                        "Choose the answer that logically fits the given physical question.\n\n### Physical Reasoning Problem:\n{instruction}\n\n### Correct Answer:\n",
                        "Pick the answer that best addresses the physical reasoning question below.\n\n### Problem:\n{instruction}\n\n### Answer:\n",
                        "Answer the physical reasoning question by selecting the best possible answer.\n\n### Question:\n{instruction}\n\n### Correct Answer:\n"
                        ],
                    "response_split": "### Answer:"
                    },
            "boolq": {
                    "description": "Template for BoolQ.",
                    "prompt_variations": [
                        "Determine whether the following statement is yes or no based on the given context.\n\n### Context:\n{instruction}\n\n### Question:\n{input}\n\n### Answer:\n",
                        "Given the context below, decide if the statement is yes or no.\n\n### Context:\n{instruction}\n\n### Statement:\n{input}\n\n### Yes or No:\n",
                        "Read the context and determine if the following statement is correct.\n\n### Background Information:\n{instruction}\n\n### Question:\n{input}\n\n### Answer (yes/no):\n",
                        "Decide if the statement is yes or no based on the context provided.\n\n### Context:\n{instruction}\n\n### Question:\n{input}\n\n### Answer:\n",
                        "Based on the given context, answer whether the statement is yes or no.\n\n### Information:\n{instruction}\n\n### Statement:\n{input}\n\n### Answer:\n",
                        "Consider the context and determine if the statement is accurate.\n\n### Context:\n{instruction}\n\n### Question:\n{input}\n\n### yes or no:\n",
                        "Using the provided context, decide if the following is yes or no.\n\n### Background:\n{instruction}\n\n### Statement:\n{input}\n\n### Answer:\n",
                        "Determine if the statement given the context is correct (yes) or not (no).\n\n### Provided Context:\n{instruction}\n\n### Statement:\n{input}\n\n### Answer (yes/no):\n",
                        "Analyze the context and decide if the statement is correct or not.\n\n### Context:\n{instruction}\n\n### Question:\n{input}\n\n### yes or no:\n",
                        "Answer whether the statement is true (yes) or false (no) based on the given context.\n\n### Information:\n{instruction}\n\n### Statement:\n{input}\n\n### Answer:\n"
                        ],
                    "response_split": "### Answer:"
                    },
            "sciq": {
                    "description": "Template for SciQ.",
                    "prompt_variations": [
                        "Below is a science question along with answer choices. Select the correct answer.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                        "You are given a science question with possible answers. Choose the correct answer.\n\n### Question:\n{instruction}\n\n### Answer Options:\n{input}\n\n### Correct Answer:\n",
                        "Choose the correct answer for the following science question from the given choices.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Selected Answer:\n",
                        "Given the science question below, pick the most accurate answer from the provided options.\n\n### Question:\n{instruction}\n\n### Options:\n{input}\n\n### Answer:\n",
                        "Read the science question and choose the best answer from the given options.\n\n### Question:\n{instruction}\n\n### Answer Choices:\n{input}\n\n### Correct Answer:\n",
                        "Select the best possible answer to the science question below.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                        "Choose the correct answer for the following science-based question.\n\n### Science Question:\n{instruction}\n\n### Options:\n{input}\n\n### Correct Answer:\n",
                        "Pick the right answer from the following multiple-choice science question.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n",
                        "Below is a science question with possible answers. Choose the correct one.\n\n### Science Question:\n{instruction}\n\n### Answer Options:\n{input}\n\n### Correct Answer:\n",
                        "Answer the following question by selecting the most appropriate science-related choice.\n\n### Question:\n{instruction}\n\n### Choices:\n{input}\n\n### Answer:\n"
                        ],
                    "response_split": "### Answer:"
                    },
            }
