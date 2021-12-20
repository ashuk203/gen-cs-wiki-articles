import pandas as pd
from tqdm import tqdm

import glob
import ujson

RAW_DATA_DIR = "data/train_raw/vlarge"
PROCESSED_DATA_DIR = "data/train_processed/vlarge"
SEP = " [SEP] "
TITLE_SEP = " [EOT] "
PREFIX = "outline"


def clean_input_text(text):
    if len(text) > 2 and text[-1] == "." and text[-2] == ".":
        return text[:-1]

    return text

def clean_title(title):
    return title.replace("_", " ")


def create_train_point(raw_obj):
    res = {}
    res['id'] = raw_obj['index']
    res['prefix'] = PREFIX 
    res['input_text'] = clean_title(raw_obj["page_title"]) + TITLE_SEP + clean_input_text(raw_obj['summary'])
    res['target_text'] = SEP.join(raw_obj['sections'])

    return res

worker_files = glob.glob(f"{RAW_DATA_DIR}/*.txt")

# Create pandas dataframe
df_rows = []

# temp_i = 0
with tqdm(total=len(worker_files)) as pbar:
    for filename in worker_files:
        with open(filename) as f:
            cur_rows = []
            for j in f.readlines():
                try:
                    cur_obj = ujson.loads(j)
                    # print(create_train_point(cur_obj))
                    # if temp_i > 5:
                    #     exit()

                    # temp_i += 1
                    
                    cur_rows.append(create_train_point(cur_obj))

                except Exception as e:
                    print(e)
                    print(j)
                    continue

            df_rows += cur_rows
            pbar.update(1)

df_indices = []
for i in range(len(df_rows) - 1, -1, -1):
    cur_row = df_rows[i]
    try:
        df_indices.append(int(cur_row["id"]))
    except Exception as e:
        print(e)
        print(type(cur_row))
        del df_rows[i]
        continue

full_df = pd.DataFrame(df_rows, index=df_indices, columns=['prefix', 'input_text', 'target_text'])
full_df.to_csv(f"{PROCESSED_DATA_DIR}/full.csv")