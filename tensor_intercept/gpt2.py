'''

GPT-2 training code

'''


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the train and validation data
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/gpt2/train.txt", # your train file
    block_size=128 # adjust according to your model's max input length
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/gpt2/valid.txt", # your validation file
    block_size=128 # adjust according to your model's max input length
)

# Define a data collator that will dynamically pad the inputs
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # GPT-2 is not a masked language model
)

training_args = TrainingArguments(
    output_dir="output", # where to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch", # evaluate after each epoch
    logging_dir="logs", # where to save the logs
    max_steps=15
)

# Create a trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Train the model
print("BEGIN")
trainer.train()
print("END")
