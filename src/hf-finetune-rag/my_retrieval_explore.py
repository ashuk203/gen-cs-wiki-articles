""" 
View intermediate results of explorer.

Consistently facing issue of "retriever collapse":
- https://github.com/huggingface/transformers/issues/9405
- https://github.com/huggingface/transformers/issues/10425
- https://discuss.huggingface.co/t/using-rag-with-local-documents/5326


Things to troubleshoot:
- have only one train example and see if retrieval collapses
- use a consolidated RAG checkpoint to see if collapse occurs

Conclusion: when only very few training examples are used, retriever does not collapse. Retrieval collapses occurs when learning rate >= 4e-6
"""

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


index_root_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_docs_index"
index_path = f"{index_root_dir}/my_knowledge_dataset_hnsw_index.faiss"
passages_path = f"{index_root_dir}/my_knowledge_dataset"

# model_root_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/models/test/kb_ft_rag"
model_root_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/models/full_run/no_def/lr5e-07_ragtok"
model_args_file = f"{model_root_dir}/model_args.pkl"
model_state_dict_file = f"{model_root_dir}/rag.model"

eval_args_file = "eval_script_args.pkl"

# full_model_out_path = "/scratch/aukey2/rag_full_models/no_inp_def_rag"
# retriever_model_out_path = "/scratch/aukey2/rag_full_models/no_inp_def_retriever"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None



def evaluate_batch_retrieval(args, rag_model, questions, retriever=None):
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title

    if retriever is None:
        retriever = rag_model.retriever

    retriever_input_ids = retriever.question_encoder_tokenizer.batch_encode_plus(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to('cpu')

    question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=rag_model.rag.generator.config.prefix,
        n_docs=rag_model.config.n_docs,
        return_tensors="pt",
    )

    # result = retriever.retrieve(
    #     question_enc_pool_output.detach().numpy(),
    #     n_docs=rag_model.config.n_docs
    # )
    # pdb.set_trace()

    all_docs = retriever.index.get_doc_dicts(result.doc_ids)
    provenance_strings = []
    for docs in all_docs:
        provenance = [strip_title(title) for title in docs["title"]]
        provenance_strings.append("\t".join(provenance))

    # pdb.set_trace()
    return provenance_strings


def get_model_kw_args():
    model_kwargs = {
        "n_docs": 5,
        "index_name": "custom",
        "passages_path": passages_path,
        "index_path": index_path
    }

    return model_kwargs


test_qs = [
    "maximum clique problem:",
    "virtual memory:",
    "robotics:",
    "logic gate:",
    "supervised learning:",
    "power method:",
    "speech coding:",
    "data recovery:",
    "simultaneous embedding:",
    "transductive learning:",
    "signal detection:",
    "envelope theorem:"
]


if __name__ == "__main__":
    with open(model_args_file, 'rb') as f:
        args = pickle.load(f)

    with open(eval_args_file, 'rb') as f:
        eval_args = pickle.load(f)

    # model = GenerativeQAModule.from_pretrained(model_root_dir)
    model_kwargs = get_model_kw_args()


    # retriever = RagRetriever.from_pretrained(
    #     "facebook/rag-sequence-base",
    #     index_name="custom",
    #     passages_path=passages_path,
    #     index_path=index_path,
    # )

    model = GenerativeQAModule(args)
    model.load_state_dict(torch.load(model_state_dict_file))
    # model.model.save_pretrained(full_model_out_path)
    # exit()
    # retriever = RagRetriever.from_pretrained(model_root_dir, **model_kwargs)
    # model = RagSequenceForGeneration.from_pretrained(model_root_dir, retriever=retriever, **model_kwargs)

    model.eval()
    # model.model.eval()
    model.model.rag.retriever.init_retrieval()
    # retriever.init_retrieval()

    # rag_model = RagSequenceForGeneration.from_pretrained(full_model_out_path)
    # retriever = RagRetriever.from_pretrained(
    #     full_model_out_path,
    #     index_name="custom",
    #     passages_path=passages_path,
    #     index_path=index_path,
    # )

    
    rag_model = model.model
    # model = model.model

    
    # prov_strs = evaluate_batch_retrieval(eval_args, rag_model, test_qs)
    prov_strs = evaluate_batch_retrieval(eval_args, rag_model, test_qs)
    for i in range(len(prov_strs)):
        q = test_qs[i]
        titles = prov_strs[i]
        print("Retrieved documents for", q, ':')

        for s in titles.split('\t'):
            print('\t(*) ', s)
