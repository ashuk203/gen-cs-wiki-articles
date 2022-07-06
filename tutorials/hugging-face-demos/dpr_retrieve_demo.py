import pdb
import torch
from transformers import RagTokenizer, BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration

kb_root_dir = "data/cs_docs_index"
dataset_path = f"{kb_root_dir}/my_knowledge_dataset"
index_path = f"{kb_root_dir}/my_knowledge_dataset_hnsw_index.faiss"

eval_args_file = "reference-code/hf-finetune-rag/eval_script_args.pkl"


# Load models
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
)

model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)


def run_retrieval(question):

    # Tokenize
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Encode
    question_hidden_states = model.question_encoder(input_ids)[0]

    # Retrieve
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")

    docs = retriever.index.get_doc_dicts(docs_dict.doc_ids)
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)

    for s in docs[0]["title"]:
        print('\t(*) ', s)
    # print(docs[0]["title"])
    # print(doc_scores)


if __name__ == '__main__':

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

    # while True:
    for question in test_qs:
        print("Default documents for", question, ':')
        # print("Enter a question for document retrieval:")
        # question = input()

        run_retrieval(question)
        print('\n')