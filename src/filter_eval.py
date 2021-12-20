import pandas as pd
from pprint import pprint

TRAIN_DATA_DIR = "data/train_processed/vlarge"
EVAL_DATA_FILE = f"{TRAIN_DATA_DIR}/eval.csv"

OUTPUT_FILE = f"{TRAIN_DATA_DIR}/eval_sub.csv"

df = pd.read_csv(EVAL_DATA_FILE).astype(str)
df = df[df["input_text"].str.contains("computer", case = False) & df["target_text"].str.contains("application", case = False)]
# df = df[df["input_text"].str.contains("algorithm", case = False)]

df.to_csv(OUTPUT_FILE)