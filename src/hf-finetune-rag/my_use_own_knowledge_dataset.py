import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import torch
from datasets import Features, Sequence, Value, load_dataset

import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

import pickle
import pdb

# from custom_qencoder import DPRAugQuestionEncoder

eos_tok = 102

logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Variables related to word2vec
embs_dict_path = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/word2vec_embs/ctx_tok2embs_dict.pickle"

with open(embs_dict_path, 'rb') as f:
    tok_to_emb = pickle.load(f)

sample_emb = list(tok_to_emb.items())[0][1]
w2v_dim = len(sample_emb["embedding"])  # 100

print("Word2vec vectors have dimension", w2v_dim)


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}



def add_w2v_embs(input_ids, embeddings):
    """
        Augments current embeddings with word2vec embeddings. 

        Arguments:
        - input_ids: tensor with shape (batch_size, max_inp_len)
        - embeddings: tensor with shape (batch_size, dim)

        Returns tensor with shape (batch_size, dim + w2v_dim)
    """

    batch_size = input_ids.shape[0]
    dim = embeddings.shape[1]

    res = torch.zeros(batch_size, 868)

    for i in range(batch_size):
        key_q = input_ids[i].tolist()
        pref_idx = key_q.index(eos_tok)

        key_q = key_q[:pref_idx]
        key_q = tuple(key_q)

        cur_w2v_emb = tok_to_emb[key_q]["embedding"]
        cur_w2v_emb = torch.tensor(cur_w2v_emb).to(device=device)
        cur_bert_emb = embeddings[i]

        aug_emb = torch.cat([cur_bert_emb, cur_w2v_emb], 0)
        res[i] = aug_emb

    return res



# TODO: change this function to use custom embeddings of context documents
def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    # input_ids = ctx_tokenizer(
    #     documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    # )["input_ids"]
    input_ids = ctx_tokenizer(
        documents["title"], documents["summary"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    pdb.set_trace()

    embeddings = add_w2v_embs(input_ids, embeddings)
    pdb.set_trace()

    return {"embeddings": embeddings.detach().cpu().numpy()}


def main(
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
):

    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(rag_example_args.csv_path), "Please provide a valid path to a csv file"

    # You can load a Dataset object this way
    dataset = load_dataset(
        "csv", data_files=[rag_example_args.csv_path], split="train", delimiter="\t", column_names=
        [
            "title",    # keyowrd
            "text", # outline
            "summary"   # first wiki paragraph
        ]
    )
    # dataset = dataset[:50]

    # More info about loading csv files in the documentation: https://huggingface.co/docs/datasets/loading_datasets.html?highlight=csv#csv-files

    # pdb.set_trace()
    # Then split the documents into passages of 100 words
    # dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc)


    # And compute the embeddings
    print("Using context encoder", rag_example_args.dpr_ctx_encoder_model_name)
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "summary": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space
    
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset")
    dataset.save_to_disk(passages_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset

    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    print("FAISS Indexing dimension: ", index_hnsw_args.d)
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)
    dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    ######################################
    logger.info("Step 3 - Load RAG")
    ######################################

    # Easy way to load the model
    print("Testing using rag model", rag_example_args.rag_model_name)
    retriever = RagRetriever.from_pretrained(
        rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset, retrieval_vector_size=868
    )
    # model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever)
    # tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    # For distributed fine-tuning you'll need to provide the paths instead, as the dataset and the index are loaded separately.
    # retriever = RagRetriever.from_pretrained(rag_example_args.rag_model_name, index_name="custom", passages_path=passages_path, index_path=index_path)
    model = RagSequenceForGeneration.from_pretrained(
        rag_example_args.rag_model_name, 
        retriever=retriever
    )
    tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    ######################################
    logger.info("Step 4 - Have fun")
    ######################################

    question = rag_example_args.question or "What does Moses' rod turn into ?"
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    logger.info("Q: " + question)
    logger.info("A: " + generated_string)


@dataclass
class RagExampleArguments:
    csv_path: str = field(
        default=str(Path(__file__).parent / "test_data" / "my_knowledge_dataset.csv"),
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default="transfer learning: transfer learning is blah blah blah",
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-base",
        metadata={"help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"},
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-single-nq-base",
        metadata={
            "help": "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or 'facebook/dpr-ctx_encoder-multiset-base'"
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        # default=768,
        default=868,    # 768 + w2v_dim
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=128,
        metadata={
            "help": "The number of bi-directional links created for every new element during the HNSW index construction."
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)
