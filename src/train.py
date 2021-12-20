import pandas as pd
from simpletransformers.t5 import T5Model

TRAIN_DATA_DIR = "data/train_processed/vlarge"
TRAIN_DATA_FILE = f"{TRAIN_DATA_DIR}/train.csv"
EVAL_DATA_FILE = f"{TRAIN_DATA_DIR}/eval.csv"


train_df = pd.read_csv(TRAIN_DATA_FILE)
eval_df = pd.read_csv(EVAL_DATA_FILE)

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "dataloader_num_workers": 1,
    "max_seq_length": 128,
    "train_batch_size": 16,
    "num_train_epochs": 1,
    "save_eval_checkpoints": True,
    "save_steps": -1,
    "num_workers": 1,
    "use_multiprocessing": False,
    "evaluate_during_training": False,
    # "evaluate_during_training_steps": 15000,
    # "evaluate_during_training_verbose": True,
    "fp16": False,

    "special_tokens_list": ["[SEP]", "[EOT]"]
}

# model = T5Model("t5", "t5-large", args=model_args)
model = T5Model("t5", "t5-base", args=model_args)
model.train_model(train_df, eval_data=eval_df)