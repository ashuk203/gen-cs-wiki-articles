export INP_DOCS_FILE=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_docs_18k/docs_40_w2v.csv
export OUT_DIR=/home/aukey2/gen-cs-wiki-articles/rag_branch/data/faiss_indexes/cs_docs_index_w2v

python3 my_use_own_knowledge_dataset.py \
    --csv_path $INP_DOCS_FILE \
    --output_dir $OUT_DIR \
    --d 868