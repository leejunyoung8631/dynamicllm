import csv


import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig




class LossCallback(TrainerCallback):
    def __init__(self, file_path="loss_log.csv", write_interval=30):
        """
        Initialize the callback with a buffer and write interval.
        """
        self.file_path = file_path
        self.write_interval = write_interval
        self.loss_buffer = []  # Temporary storage for loss data
        
        # Ensure the directory exists
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write the header to the CSV file
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "loss"])  # CSV Header

    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log loss every step in memory and write to the file every `write_interval` steps.
        """
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            step = state.global_step
            
            # Add to buffer
            self.loss_buffer.append((step, loss))
            
            # Write to file if buffer reaches the interval
            if len(self.loss_buffer) >= self.write_interval:
                self._flush_buffer()
                
    
    def _flush_buffer(self):
        """
        Write buffered loss data to the CSV file and clear the buffer.
        """
        if not self.loss_buffer:
            return  # Nothing to write
        
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.loss_buffer)
        
        print(f"Flushed {len(self.loss_buffer)} steps of loss data to {self.file_path}")
        self.loss_buffer = []  # Clear the buffer
        

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Ensure all remaining data is written when training ends.
        """
        self._flush_buffer()
        print("Training finished. All buffered data has been written.")





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






def create_trainer(
        model, tokenizer, train_data, val_data,
        epochs, outdir,
        show_progress, load_best_model,
        args,
        custom_eval_save_step=None):
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
        save_steps=None
        save_total_limit=None
        load_best_model_at_end=False
        save_strategy="no"
        
    
    save_strategy = "steps"
    save_steps = 400
    

    callbacks = []
    # eval_callback = LossCallback()
    # callbacks.append(eval_callback)

    trainer = Trainer(
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