"""
A script creating a RAG checkpoint from a generator and a question encoder checkpoints.
"""
import sys
sys.path.insert(1, '/home/aukey2/gen-cs-wiki-articles/rag_branch/src')

from custom_qencoder import DPRAugQuestionEncoder

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, RagConfig, RagSequenceForGeneration, RagTokenForGeneration, DPRConfig, BartForConditionalGeneration

import pdb


def consolidate(
    model_type,
    generator_name_or_path: str,
    question_encoder_name_or_path: str,
    dest_dir: Path,
    config_name_or_path: str = None,
    generator_tokenizer_name_or_path: str = None,
    question_encoder_tokenizer_name_or_path: str = None,
):

    if config_name_or_path is None:
        config_name_or_path = "facebook/rag-token-base" if model_type == "rag_token" else "facebook/rag-sequence-base"

    if generator_tokenizer_name_or_path is None:
        generator_tokenizer_name_or_path = generator_name_or_path

    if question_encoder_tokenizer_name_or_path is None:
        question_encoder_tokenizer_name_or_path = question_encoder_name_or_path

    model_class = RagTokenForGeneration if model_type == "rag_token" else RagSequenceForGeneration

    # Save model.
    rag_config = RagConfig.from_pretrained(config_name_or_path)
    gen_config = AutoConfig.from_pretrained(generator_name_or_path)
    question_encoder_config = AutoConfig.from_pretrained(question_encoder_name_or_path)

    rag_config.generator = gen_config
    rag_config.question_encoder = question_encoder_config

    """
    TODO: @Ashu the problem is that the question_encoder_name_or_path is not recognizing your custom question encoder that you have defined, you may need to explicity specify this

    Structure of variable
    rag_model: RagTokenForGeneration
        - rag: RagModel
            - generator: models.bart.modeling_bart.BartForConditionalGeneration
            - retriever: distributed_ray_retriever.RagRayDistributedRetriever
            - question_encoder: custom_qencoder.DPRAugQuestionEncoder
        - question_encoder: custom_qencoder.DPRAugQuestionEncoder

    Possible solutions:
    1) Download source code and try directly modifying RAG class(es). May need to import a lot of things...
    2) Figure out which variables can override the default class assumption of  
    """
    # Solution 2
    config  = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder_model = DPRAugQuestionEncoder(config)


    rag_model = model_class.from_pretrained_question_encoder_generator(
        None,
        generator_name_or_path, # facebook/bart-large
        config=rag_config,
        question_encoder_model=question_encoder_model # question_encoder_name_or_path,  # /scratch/aukey2/rag-ft-models/blank/aug_qencoder
    )
    # pdb.set_trace()

    # Saving qencoder component seperately because it is causing issues
    rag_model.save_pretrained(dest_dir)
    rag_model.rag.generator.save_pretrained(dest_dir / "gener")
    rag_model.rag.question_encoder.save_pretrained(dest_dir / "q_encoder")
    # rag_model.question_encoder.save_pretrained(dest_dir / "q_encoder")


    # Sanity check (reloading from individual parts)
    temp_gener = BartForConditionalGeneration.from_pretrained(dest_dir / "gener")
    temp_q_enc = DPRAugQuestionEncoder.from_pretrained(dest_dir / "q_encoder")
    # temp = model_class.from_pretrained(dest_dir)

    temp = model_class.from_pretrained_question_encoder_generator(
        None,
        None,
        config=rag_config,
        generator_model=temp_gener,
        question_encoder_model=temp_q_enc
    )
    # pdb.set_trace()

    print("Testing load model from", dest_dir)

    # Save tokenizers.
    
    # additional_special_tokens=["[SECTSEP]", "[DEF]"]
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_name_or_path)
    gen_tokenizer.save_pretrained(dest_dir / "generator_tokenizer/")

    # additional_special_tokens=["[SECTSEP]", "[DEF]"]
    question_encoder_tokenizer = AutoTokenizer.from_pretrained(question_encoder_tokenizer_name_or_path)
    question_encoder_tokenizer.save_pretrained(dest_dir / "question_encoder_tokenizer/")
    print("Saving tokenizers! to", dest_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token"],
        required=True,
        type=str,
        help="RAG model type: rag_sequence, rag_token",
    )
    parser.add_argument("--dest", type=str, required=True, help="Path to the output checkpoint directory.")
    parser.add_argument("--generator_name_or_path", type=str, required=True, help="Generator model identifier")
    parser.add_argument(
        "--question_encoder_name_or_path", type=str, required=True, help="Question encoder model identifier"
    )

    parser.add_argument(
        "--generator_tokenizer_name_or_path",
        type=str,
        help="Generator tokenizer identifier, if not specified, resolves to ``generator_name_or_path``",
    )
    parser.add_argument(
        "--question_encoder_tokenizer_name_or_path",
        type=str,
        help="Question encoder tokenizer identifier, if not specified, resolves to ``question_encoder_name_or_path``",
    )
    parser.add_argument(
        "--config_name_or_path",
        type=str,
        help="Identifier of the model config to use, if not provided, resolves to a base config for a given ``model_type``",
    )

    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(exist_ok=True)

    consolidate(
        args.model_type,
        args.generator_name_or_path,
        args.question_encoder_name_or_path,
        dest_dir,
        args.config_name_or_path,
        args.generator_tokenizer_name_or_path,
        args.question_encoder_tokenizer_name_or_path,
    )
