"""
    Inference is pretty bad for learning rate <= 8e-6
"""
import sys
sys.path.insert(1, '/home/aukey2/gen-cs-wiki-articles/rag_branch/src')

from custom_qencoder import DPRAugQuestionEncoder


import torch
import pickle
from finetune_rag import GenerativeQAModule

import pdb
from my_retrieval_explore import test_qs

from transformers import BartForConditionalGeneration, RagTokenForGeneration, RagConfig, RagTokenizer
from transformers.integrations import is_ray_available


if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever


# One of {"rag_token", "rag_base"}
model_type = "rag_token"

model_root_dir = "/scratch/aukey2/rag-ft-models/test/w2v_aug_qenc_kb/save_pretrained_ashu"
model_args_file = f"{model_root_dir}/model_args.pkl"
model_state_dict_file = f"{model_root_dir}/rag.model"


def get_token_id_tensor(tokenizer, txt):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor


def run_model(model, inp_txt):
    # inp_token_ids = get_token_id_tensor(model.tokenizer, inp_txt)
    inputs = model.tokenizer(inp_txt, return_tensors="pt")

    # outputs is of type 
    # <class 'transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput'>
    # outputs = model(input_ids=inputs["input_ids"])
    # print("Heyyyy, inference inp ids:", inputs["input_ids"])
    outputs = model.model.generate(input_ids=inputs["input_ids"])
    output_texts = model.tokenizer.batch_decode(outputs)

    return output_texts


def run_rag_model(model, tokenizer, inp_txt):
    # inp_token_ids = get_token_id_tensor(model.tokenizer, inp_txt)
    inputs = tokenizer(inp_txt, return_tensors="pt")

    # outputs is of type 
    # <class 'transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput'>
    # outputs = model(input_ids=inputs["input_ids"])
    # print("Heyyyy, inference inp ids:", inputs["input_ids"])
    outputs = model.generate(input_ids=inputs["input_ids"])
    output_texts = tokenizer.batch_decode(outputs)

    return output_texts


def main_sample(model):
    # Run the model on some test input
    inp_txt1 = 'The costs could hit JD Sports\' expansion plans, he added, which could mean fewer extra jobs. Thanasi Kokkinakis backed by Tennis Australia president Steve Healy Thanasi Kokkinakis deserves kudos rather than criticism for his behaviour. Thanasi Kokkinakis has been the collateral damage in the recent storm around his friend Nick Kyrgios and deserves kudos rather than criticism for his own behaviour, according to Tennis Australia president Steve Healy.'

    inp_txt2 = 'UN Chief Says There Is No Military Solution in Syria Secretary-General Ban Ki-moon says his response to Russia\'s stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people. The U.N. chief again urged all parties, including the divided U.N. Security Council, to unite and support inclusive negotiations to find a political solution. Ban told a news conference Wednesday that he plans to meet with foreign ministers of the five permanent council nations - the U.S., Russia, China, Britain and France - on the sidelines of the General Assembly\'s ministerial session later this month to discuss Syria.'

    inp_txt3 = 'But one group of former street children have found a way to learn a skill and make a living. "I was shot in Joburg" is a non-profit studio that teaches homeless youngsters how to take photographs of their neighbourhood and make a profit from it. BBC News went to meet one of the project\'s first graduates. JD Sports boss says higher wages could hurt expansion JD Sports Executive Chairman Peter Cowgill says a higher minimum wage for UK workers could mean "more spending power in the pockets of potential consumers." But that spending power is unlikely to outweigh the higher labour costs at his firm, he says.'

    out_txt1 = run_model(model, inp_txt1)
    out_txt2 = run_model(model, inp_txt2)
    out_txt3 = run_model(model, inp_txt3)

    print(out_txt1)
    print(out_txt2)
    print(out_txt3)


def main_keywords(model):
    out_txt1 = run_model(model, "machine learning")
    out_txt2 = run_model(model, "data mining")
    out_txt3 = run_model(model, "data structures")

    print(out_txt1)
    print(out_txt2)
    print(out_txt3)


def main_cs_test(model):
    inps = [
        "cryptographic hash function: A cryptographic hash function (CHF) is a mathematical algorithm that maps data of an arbitrary size (often called the \"message\") to a bit array of a fixed size (the \"hash value\", \"hash\", or \"message digest\"). It is a one-way function, that is, a function for which it is practically infeasible to invert or reverse the computation. Ideally, the only way to find a message that produces a given hash is to attempt a brute-force search of possible inputs to see if they produce a match, or use a rainbow table of matched hashes. Cryptographic hash functions are a basic tool of modern cryptography.",
        "card sorting: Card sorting is a technique in user experience design in which a person tests a group of subject experts or users to generate a dendrogram (category tree) or folksonomy. It is a useful approach for designing information architecture, workflows, menu structure, or web site navigation paths.",
        "information leakage: Information leakage happens whenever a system that is designed to be closed to an eavesdropper reveals some information to unauthorized parties nonetheless. In other words: Information leakage occurs when secret information correlates with, or can be correlated with, observable information. For example, when designing an encrypted instant messaging network, a network engineer without the capacity to crack encryption codes could see when messages are transmitted, even if he could not read them.",
        "database security: Database security concerns the use of a broad range of information security controls to protect databases (potentially including the data, the database applications or stored functions, the database systems, the database servers and the associated network links) against compromises of their confidentiality, integrity and availability. It involves various types or categories of controls, such as technical, procedural/administrative and physical."
    ]


    for inp in inps:
        outline = run_rag_model(model, inp)
        print(outline)


def main_inps(model, inps):
    for inp in inps:
        outline = run_model(model, inp)
        print('(*)  ', inp, outline)
        # print(outline)
        print('\n')


def main_loop(model):
    while True:
        print("Enter input to RAG model: ")
        inp = input()
        out = run_model(model, inp)


        print(out)
        print('-*-' * 20)


def load_model():
    with open(model_args_file, 'rb') as f:
        args = pickle.load(f)

    # model = GenerativeQAModule.from_pretrained(model_root_dir)
    model = GenerativeQAModule(args)
    model.load_state_dict(torch.load(model_state_dict_file))


def load_model_sp():
    """
        Load model that was saved using 'save_pretrained'
    """
    global model_root_dir
    model_root_dir += '/'

    # Modify if more class type support is needed
    model_class = RagTokenForGeneration

    config_name_or_path = "facebook/rag-token-base" if model_type == "rag_token" else "facebook/rag-sequence-base"
    rag_config = RagConfig.from_pretrained(config_name_or_path)

    temp_gener = BartForConditionalGeneration.from_pretrained(model_root_dir + "gener")
    temp_q_enc = DPRAugQuestionEncoder.from_pretrained(model_root_dir + "q_encoder")
    temp_retriev = RagRayDistributedRetriever.from_pretrained(model_root_dir + "retriev", actor_handles=[])
    # temp = model_class.from_pretrained(dest_dir)

    temp = model_class.from_pretrained_question_encoder_generator(
        None,
        None,
        config=rag_config,
        generator_model=temp_gener,
        question_encoder_model=temp_q_enc,
        retriever_model=temp_retriev
    )

    return temp


if __name__ == '__main__':

    # Load saved model
    model = load_model_sp()
    model.eval()

    tokenizer = RagTokenizer.from_pretrained("/scratch/aukey2/rag-ft-models/blank/w2v_aug_rag")

    main_cs_test(model, tokenizer)
    # main_keywords(model)
    # main_sample(model)
    # main_loop(model)
    # main_inps(model, test_qs)



