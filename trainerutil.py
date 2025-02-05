import os
import csv
import numpy as np

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig

from dyllm_model import DyLLM



# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         We override the compute_loss method to use a custom loss.
#         """
#         # Forward pass
#         outputs = model(**inputs)
        
#         # target
#         labels = inputs.get('labels') 
#         b, n = labels.shape
#         origin_loss = outputs.get("loss")
#         origin_loss = origin_loss.reshape(b, -1).mean(dim=-1)
        
#         # first layer feature
#         # this will predict the loss 
#         hidden = outputs.get('hidden_states')[0]
        
#         pred = model.predictor(hidden)
#         loss = LossPredLoss(pred, origin_loss)
        
#         # If we want to return outputs (for logging or other use),
#         # we return (loss, outputs). Otherwise we just return the loss.
#         return (loss, outputs) if return_outputs else loss



class LossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # We'll store the values in this list
        self.losses = []
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Access the model
        model = kwargs.get('model', None)
        
        mask_loss = getattr(model, "mask_loss", 0)
        distill_loss = getattr(model, "distill_loss", 0)  
        ppl_loss = getattr(model, "ppl_loss", 0)  
        self.losses.append([mask_loss, distill_loss, ppl_loss])
        
        current_step = state.global_step
        path = os.path.join(args.output_dir, "lossfile.npy")
        if current_step % 100 == 0:
            path = os.path.join(args.output_dir, "lossfile.npy")
            np.save(path, np.array(self.losses))
                
        return control    
    
    
    def on_train_end(
        self, 
        args, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        path = os.path.join(args.output_dir, "lossfile.npy")
        np.save(path, np.array(self.losses))

        return control  




# Iteration to mask module
class IterationCheckTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        # steps
        current_step = self.state.global_step
        total_steps = self.state.max_steps
        
        if hasattr(model, "diff_mask"):
            model.diff_mask.iteration = current_step
            model.diff_mask.whole_iter = total_steps

        # Now call the default implementation
        return super().training_step(model, inputs)
    


Trainer_Mapping = {
    "iter" : IterationCheckTrainer,
    "normal" : Trainer,
}

Callback_Mapping = {
    "loss" : LossCallback,
}



def create_trainer(
        model, tokenizer, train_data, val_data,
        epochs, outdir,
        show_progress, load_best_model,
        args,
        custom_eval_save_step = None,
        custom_trainer = None,
        custom_callback = None):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    fp16_flag = True
    bf16_flag = False
    if args.use_bfloat:
        model = model.bfloat16()
        fp16_flag = False
        bf16_flag = True

    if load_best_model:
        eval_steps=200
        evaluation_strategy="steps"
        save_steps=200
        save_total_limit=20
        load_best_model_at_end=True
        save_strategy="steps"
    else:
        eval_steps=None
        evaluation_strategy="no"
        save_steps=400
        save_total_limit=None
        load_best_model_at_end=False
        save_strategy="steps"
        
        
    if isinstance(model, DyLLM):
        print("Automatically DyLLM use IterationTrainer")
        custom_trainer = "iter"    
        
    
    # custom trainer (optional)
    trainer_class = None
    if custom_trainer is not None:
        if custom_trainer not in Trainer_Mapping.keys():
            print(f"{custom_trainer} is not supported")
            print("Supported Trainer")
            print(Trainer_Mapping.keys())
        trainer_class = Trainer_Mapping[custom_trainer]
    else: 
        trainer_class = Trainer_Mapping["normal"]
        
    
    # add callback (optional)
    callbacks = []
    if custom_callback is not None:
        for callback in custom_callback:
            if callback not in Callback_Mapping.keys():
                print(f"{callback} is not supported")
                print("Supported Trainer")
                print(Callback_Mapping.keys())
            callbacks.append(Callback_Mapping[callback])
            
    

    trainer = trainer_class(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=callbacks,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=args.learning_rate,
            fp16=fp16_flag,
            bf16=bf16_flag,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=outdir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            # report_to="wandb",
            report_to="none",
            run_name=args.output_dir.split("/")[-1],
            # metric_for_best_model="{}_loss".format("yahma/alpaca-cleaned"),
            disable_tqdm=(not show_progress),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    return trainer