from transformers import TrainingArguments

# Set training parameters
training_params = TrainingArguments(
    output_dir="results",  # The output directory where the model predictions and checkpoints will be written.
    num_train_epochs=1,  # Total number of training epochs to perform.
    per_device_train_batch_size=4,  # The batch size per GPU/TPU core/CPU for training.
    gradient_accumulation_steps=1,  #  Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
    optim="paged_adamw_32bit",
    save_steps=100,  # Number of updates steps before two checkpoint saves.
    logging_steps=100,  # Number of update steps between two logs.
    learning_rate=2e-4,  # The initial learning rate for Adam.
    weight_decay=0.001,  # The weight decay to apply (if not zero).
    fp16=False,  # Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
    bf16=False,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    max_grad_norm=0.3,  # Maximum gradient norm (for gradient clipping).
    max_steps=-1,  # If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.
    warmup_ratio=0.03,
    group_by_length=True,
    seed=42,
    lr_scheduler_type="cosine"  # "constant",
    # report_to="tensorboard"
)