from functools import partial
from const import END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, PROMPT_WITH_INPUT_FORMAT, PROMPT_NO_INPUT_FORMAT, RESPONSE_KEY_NL
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/pythia-2.8b"
cache_dir="bala_llm/models"

model  = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer  = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
model.resize_token_embeddings(len(tokenizer))
conf = model.config
max_length = None
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
    max_length = getattr(model.config, length_setting, None)
    if max_length:
        print(f"Found max lenth: {max_length}")
        break
if not max_length:
    max_length = 1024
    print(f"Using default max length: {max_length}")

seed = 42
test_size = 200
def format_record(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")
    
    if context:
        rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
    else:
        rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
    return rec

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

_preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
remove_columns=["instruction", "context", "response", "text", "category"]

dataset = (
    load_dataset("databricks/databricks-dolly-15k")["train"]
        .map(format_record)
        .map(_preprocessing_function, batched=True, remove_columns=remove_columns)
        .filter(lambda rec: len(rec["input_ids"]) < max_length)
        .shuffle(seed=seed)
)


split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
from transformers import Trainer, TrainingArguments

output_dir = "trained_model"

training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=True,
        learning_rate=5e-4,
        num_train_epochs=0.05,
        deepspeed="config.json",
        gradient_checkpointing=False,
        logging_dir=f"{output_dir}/runs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=10,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=False,
        warmup_steps=0,
    )
from transformers import DataCollatorForLanguageModeling
import numpy as np
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("./custom_model/v2")



# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# mm = AutoModelForCausalLM.from_pretrained("./custom_model/v1")
# tt = AutoTokenizer.from_pretrained("./custom_model/v1")
# tt.decode(mm.generate(**tt("generate json for movie reviews", return_tensors="pt"))[0])
# tt("generate json for movie reviews", return_tensors="pt")
# model_name = "EleutherAI/pythia-1b"
# cache_dir="bala_llm/models"

# model  = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# tokenizer  = AutoTokenizer.from_p
# tt.decode(res[0])
