import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA_DIR = "data/train_processed/vlarge"
LABELED_DATA_FILE = f"{TRAIN_DATA_DIR}/full.csv"
TRAIN_DATA_FILE = f"{TRAIN_DATA_DIR}/train.csv"
EVAL_DATA_FILE = f"{TRAIN_DATA_DIR}/eval.csv"

TEST_DATA_PROP = 0.2

full_df = pd.read_csv(LABELED_DATA_FILE)
full_df = full_df[full_df['target_text'].notna()]

train_df, eval_df = train_test_split(full_df, test_size=TEST_DATA_PROP)
train_df.to_csv(TRAIN_DATA_FILE)
eval_df.to_csv(EVAL_DATA_FILE)


print(train_df.shape[0], train_df.columns)
print(eval_df.shape[0], eval_df.columns)