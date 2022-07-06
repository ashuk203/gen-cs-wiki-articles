# --question_encoder_name_or_path /home/aukey2/gen-cs-wiki-articles/rag_branch/models/test/aug_qencoder_test \


python3 consolidate_rag_checkpoint.py \
    --model_type rag_token \
    --generator_name_or_path facebook/bart-large \
    --question_encoder_name_or_path /scratch/aukey2/rag-ft-models/blank/aug_qencoder \
    --question_encoder_tokenizer_name_or_path facebook/dpr-question_encoder-single-nq-base \
    --dest /scratch/aukey2/rag-ft-models/blank/w2v_aug_rag