""" Evaluation script for RAG models."""

import argparse
import ast
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils_rag import exact_match_score, f1_score  # noqa: E402 # isort:skip

from finetune_rag import GenerativeQAModule
import json
import pickle

import pdb


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()

# torch.cuda.empty_cache()


index_root_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_docs_index"
index_path = f"{index_root_dir}/my_knowledge_dataset_hnsw_index.faiss"
passages_path = f"{index_root_dir}/my_knowledge_dataset"

model_root_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/models/full_run/no_def/lr2e-07_ragtoken"
model_args_file = f"{model_root_dir}/model_args.pkl"
model_state_dict_file = f"{model_root_dir}/rag.model"

train_data_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_train_data_no_inp_def"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart"],
        type=str,
        help="RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the model_name_or_path",
    )
    parser.add_argument(
        "--index_name",
        default=None,
        choices=["exact", "compressed", "legacy"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--index_path",
        default=None,
        type=str,
        help="Path to the retrieval index",
    )
    parser.add_argument("--n_docs", default=5, type=int, help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval"],
        default="e2e",
        type=str,
        help="Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates precision@k.",
    )
    parser.add_argument("--k", default=1, type=int, help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help="Format of the gold data file"
        "qa - a single line in the following format: question [tab] answer_list"
        "ans - a single line of the gold file contains the expected answer string",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")

    parser.add_argument(
        "--print_predictions",
        # default=True,
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args




def evaluate_batch_e2e(args, rag_model, questions):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs_dict.input_ids.to(args.device)
        attention_mask = inputs_dict.attention_mask.to(args.device)
        outputs = rag_model.generate(  # rag_model overwrites generate
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if args.print_predictions:
            for q, a in zip(questions, answers):
                out_ln = "Q: {} - A: {}".format(q, a)
                logger.info(out_ln)
                print(out_ln)

        return answers


test_qs = [
        "cryptographic hash function: A cryptographic hash function (CHF) is a mathematical algorithm that maps data of an arbitrary size (often called the \"message\") to a bit array of a fixed size (the \"hash value\", \"hash\", or \"message digest\"). It is a one-way function, that is, a function for which it is practically infeasible to invert or reverse the computation. Ideally, the only way to find a message that produces a given hash is to attempt a brute-force search of possible inputs to see if they produce a match, or use a rainbow table of matched hashes. Cryptographic hash functions are a basic tool of modern cryptography.",
        "card sorting: Card sorting is a technique in user experience design in which a person tests a group of subject experts or users to generate a dendrogram (category tree) or folksonomy. It is a useful approach for designing information architecture, workflows, menu structure, or web site navigation paths.",
        "information leakage: Information leakage happens whenever a system that is designed to be closed to an eavesdropper reveals some information to unauthorized parties nonetheless. In other words: Information leakage occurs when secret information correlates with, or can be correlated with, observable information. For example, when designing an encrypted instant messaging network, a network engineer without the capacity to crack encryption codes could see when messages are transmitted, even if he could not read them.",
        "database security: Database security concerns the use of a broad range of information security controls to protect databases (potentially including the data, the database applications or stored functions, the database systems, the database servers and the associated network links) against compromises of their confidentiality, integrity and availability. It involves various types or categories of controls, such as technical, procedural/administrative and physical."
    ]

if __name__ == "__main__":
    with open(model_args_file, 'rb') as f:
        args = pickle.load(f)

    # Load RAG model
    e2e_args = get_args()

    model = GenerativeQAModule(args)
    model.load_state_dict(torch.load(model_state_dict_file))

    model.eval()
    rag_model = model.model.cuda()
    
    print(e2e_args)


    ##  Run RAG model on test data   ##

    with open(f"{train_data_dir}/test.source") as f:
        inps = f.readlines()

    print("Generating outlines for", len(inps), "examples")
    model_outlines = []

    # Generate outlines in batches
    q_batch_size = 8
    for i in tqdm(range(0, len(inps), q_batch_size)):
        cur_batch_qs = inps[i: i + q_batch_size]
        cur_outls = evaluate_batch_e2e(e2e_args, rag_model, cur_batch_qs)
        model_outlines += cur_outls

    with open(f"{train_data_dir}/test.model_predicted", "w") as f:
        for outl in model_outlines:
            f.write(outl + '\n')
