export MODEL_NAME_OR_PATH=facebook/rag-sequence-nq

<< EXPLANATION
     --model_name_or_path $MODEL_NAME_OR_PATH \ # model name or path of the model we're evaluating
     --model_type rag_sequence \ # RAG model type (rag_token or rag_sequence)
     --evaluation_set output/biencoder-nq-dev.questions \ # an input dataset for evaluation
     --gold_data_path poutput/biencoder-nq-dev.pages \ # a dataset containing ground truth answers for samples from the evaluation_set
     --predictions_path output/retrieval_preds.tsv  \ # name of file where predictions will be stored
     --eval_mode retrieval \ # indicates whether we're performing retrieval evaluation or e2e evaluation
     --k 1 # parameter k for the precision@k metric
EXPLANATION

python3 eval_rag.py \
     --model_name_or_path $MODEL_NAME_OR_PATH \
     --model_type rag_sequence \
     --evaluation_set output/biencoder-nq-dev.questions \
     --gold_data_path poutput/biencoder-nq-dev.pages \
     --predictions_path output/retrieval_preds.tsv  \
     --eval_mode retrieval \
     --k 1