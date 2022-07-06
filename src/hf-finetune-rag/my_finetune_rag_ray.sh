: << 'COMMENT'
Sample script to finetune RAG using Ray for distributed retrieval.
Best performing model in full_run/no_def/lr5e-07_ragtok

Run times:
- Epoch 0:  98%|███████████████▌| 7686/7879 [57:26<01:26,  2.23it/s, loss=71.8, v_num=78]
- Generating outputs for test data
100%|██████████████████████████████████████████████| 1729/1729 [06:51<00:00,  4.20it/s]
COMMENT

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# export DATA_DIR=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_train_data_no_inp_def
export DATA_DIR=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/training_data/cs_train_data_w2v

# export OUTPUT_DIR=/scratch/aukey2/rag-ft-models/full_run/w2v_one_sent/lr5e-07_ragtok
export OUTPUT_DIR=/scratch/aukey2/rag-ft-models/full_run/w2v_kb_qenc

# export MODEL_NAME_OR_PATH=facebook/rag-sequence-base
# export MODEL_NAME_OR_PATH=facebook/rag-token-base
export MODEL_NAME_OR_PATH=/scratch/aukey2/rag-ft-models/blank/w2v_aug_rag

# export MODEL_TYPE=rag_sequence
export MODEL_TYPE=rag_token

# export MODEL_NAME_OR_PATH=/home/aukey2/gen-cs-wiki-articles/rag_branch/models/test/rag_tok_test
# export MODEL_NAME_OR_PATH=/home/aukey2/gen-cs-wiki-articles/rag_branch/models/test/aug_rag_test

export DOC_INDEX_PATH=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/faiss_indexes/cs_docs_index_w2v/my_knowledge_dataset_hnsw_index.faiss
export DOC_PASSAGES_PATH=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/faiss_indexes/cs_docs_index_w2v/my_knowledge_dataset

# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag_ray.sh --help to see all the possible options

# --do_predict \
# --learning_rate 3e-05 \

python3 finetune_rag.py \
    --distributed_retriever ray \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --index_name custom \
    --index_path $DOC_INDEX_PATH \
    --passages_path $DOC_PASSAGES_PATH \
    --model_type $MODEL_TYPE \
    --gpus 1  \
    --profile \
    --do_train \
    --n_val -1  \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 5e-07 \
    --num_train_epochs 1 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    # \
    # &> last-run-stdout.log

# Stop the Ray cluster.
ray stop
