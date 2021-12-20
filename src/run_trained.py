from common import HEADING_SEP

from simpletransformers.t5 import T5Model
import pandas as pd
from pprint import pprint

TRAIN_DATA_DIR = "data/train_processed/vlarge"
# EVAL_DATA_FILE = f"{TRAIN_DATA_DIR}/eval_sub.csv"
EVAL_DATA_FILE = "data/train_processed/cs_defs.csv"
TRAINED_MODEL_FILE = "outputs"


OUTPUT_FILE = "data/generated_outlines_csdefs.txt"

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "eval_batch_size": 128,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "use_multiprocessing": False,
    "do_sample": True,
    # "do_sample": False,
    "max_length": 50,
    # "max_length": 100,


    "special_tokens_list": ["[SEP]", "[EOT]"],

    # Best for t-5 base
    # "num_beams": 4,
    # "num_return_sequences": 2,
    # "repetition_penalty": 1.0,
    # "top_k": 20,
    # "top_p": 0.7,

    # Experimental
    "num_return_sequences": 3,
    "num_beams": 4,
    "repetition_penalty": 5.0,
    "top_k": 20,
    "top_p": 0.8,
    "length_penalty": 0.0

}

model = T5Model("t5", TRAINED_MODEL_FILE, args=model_args)

df = pd.read_csv(EVAL_DATA_FILE).astype(str)
preds = model.predict(
    ["outline: " + description for description in df["input_text"].tolist()]
)


# questions = df["target_text"].tolist()

def consolidate_headings(outputs):
    """
        Returns union of all generated heading sequences

        Arguments:
        - outputs: array of sequences
    """
    all_headings = set()

    for headings_raw in outputs:
        cur_headings = str(headings_raw).split(HEADING_SEP)
        all_headings.update(cur_headings)

    return HEADING_SEP.join(all_headings)

with open(OUTPUT_FILE, "w") as f:
    show_seperate_headings = False

    for i, desc in enumerate(df["input_text"].tolist()):
        # pprint(desc)
        # pprint(preds[i])
        # print()

        f.write(str(desc) + "\n\n")

        # f.write("Real headings:\n")
        # f.write(questions[i] + "\n\n")

        f.write("Generated headings:\n")

        if not show_seperate_headings and model_args["num_return_sequences"] > 1:

            consol_pred = consolidate_headings(preds[i])
            f.write(consol_pred + '\n')
        else:
            for pred in preds[i]:
                if model_args["num_return_sequences"] > 1:
                    f.write("(*) ")

                f.write(str(pred))

                if model_args["num_return_sequences"] > 1:
                    f.write("\n")
        f.write("\n________________________________________________________________________________\n")