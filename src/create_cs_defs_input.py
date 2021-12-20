import pandas as pd
from common import PREFIX, TITLE_SEP

INPUT_FILE = "data/generated_definition_for_cs_term.txt"
OUTPUT_FILE = "data/train_processed/cs_defs.csv"

NUM_SAMPLES = 500

df_rows = []

def create_input_row(raw_text):
    row = {}
    row["input_text"] = raw_text.replace(" [DEF] ", TITLE_SEP)
    row["prefix"] = PREFIX

    return row


with open(INPUT_FILE, "r") as f:
    lines = f.readlines()
    lines = lines[:NUM_SAMPLES]


df_rows = [create_input_row(line) for line in lines]

df = pd.DataFrame(df_rows)
df.to_csv(OUTPUT_FILE)



    
