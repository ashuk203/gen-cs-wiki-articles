This module predicts the outline for an article given its title and a short sentence description about it. The README file contains details about how to setup and run the code. 

For a research report, please refer to [Research-report.md](Research-report.md)

# Setup

Below steps were tested using Python version 3.8.12 and pip 21.2.4. If you run into errors when following the below steps, please try using these exact versions. Also be sure to bave GPUs available. This code was tested on Owl3 with

```
NVIDIA-SMI 470.103.01
Driver Version: 470.103.01
CUDA Version: 11.4
```

1. Install python dependencies

   ```
   pip install -r src/hf-finetune-rag/requirements.txt
   ```

2. Download the extract all of the following files (into a location with plenty of memory):
    * data.zip [(download from Google Drive)](https://drive.google.com/file/d/1JM2wofuZJA-Z9d7h68DbYQAVx7SaB-jo/view?usp=sharing)
    * rag-ft-models.zip [~20G] [(download from Google Drive)](https://drive.google.com/file/d/1htTRYSFnHxLO2CypaC8YQvykW2m_BFRa/view?usp=sharing)
    * word2vec.model.zip [(download from Google Drive)](https://drive.google.com/file/d/1Wey6ZCVtkO6JwrDYcmII5kMjaMkSCqfA/view?usp=sharing)
    * article_jsons.txt.zip [(download from Google Drive)](https://drive.google.com/file/d/1VFocNoAlg4ehshXBDcOwxxiAWurgGVtR/view?usp=sharing)

    Keep a note of these folder paths as you may need to certain fields in the below steps with these paths. 

3. Rename the `rag-ft-models` to `models`. Create an empty directory called `word2vec_embs`

It is very important to also include an overall breakdown of your repo's file structure. Let people know what is in each directory and where to look if they need something specific. This will also let users know how your repo needs to structured so that your module can work properly

## File structure

After downloading all of the required files, your repo structure should looks something like

```
ashutosh-ukey-gen-article-outline-rag/
├── models
│   ├── blank
│   │   ├── aug_qencoder
│   │   └── w2v_aug_rag
│   ├── full_run
│   │   ├── no_def_ft_rags
│   │   ├── one_sent
│   │   ├── w2v_kb_qenc
│   │   └── w2v_one_sent
│   ├── tests
├── data
│   ├── cs_docs_18k
│   │   ├── cs-wiki-docs.tsv
│   │   ├── docs_40_w2v.csv
│   │   ├── docs_45.csv
│   │   └── docs.csv
│   ├── faiss_indexes
│   │   ├── cs_doc_index_test
│   │   ├── cs_docs_index
│   │   └── cs_docs_index_w2v
│   ├── training_data
│   │   ├── cs_train_data
│   │   ├── cs_train_data_no_inp_def
│   │   ├── cs_train_data_old
│   │   ├── cs_train_data_subset
│   │   ├── cs_train_data_w2v
│   │   ├── cs_train_data_w2v_subset
│   │   └── cs_train_first_sent
│   ├── tutorials
│   └── word2vec_embs
│       ├── ctx_tok2embs_dict.pickle
│       ├── tok2embs_dict.pickle
│       └── w2v_embs_dict.pickle
├── documentation-pics
├── README.md
├── Research-report.md
├── src
│   ├── custom_qencoder.py
│   ├── data-prep
│   │   ├── connect_tok_embs.py
│   │   ├── create_cs_docs.py
│   │   ├── create_index_and_split.py
│   │   ├── create_train_data.py
│   │   ├── create_train_subset.py
│   │   ├── explore_embs.py
│   │   ├── explore_tok2embs.py
│   │   ├── split_train_data.py
│   │   ├── store_embeddings.py
│   │   └── strip_defs_train.py
│   ├── hf-finetune-rag
│   │   ├── callbacks_rag.py
│   │   ├── consolidate_rag_checkpoint.py
│   │   ├── distributed_pytorch_retriever.py
│   │   ├── distributed_ray_retriever.py
│   │   ├── eval_rag.py
│   │   ├── eval_script_args.pkl
│   │   ├── finetune_rag.py
│   │   ├── finetune_rag_ray.sh
│   │   ├── finetune_rag.sh
│   │   ├── __init__.py
│   │   ├── last-run-stdout.log
│   │   ├── lightning_base.py
│   │   ├── lightning_logs
│   │   ├── my_calc_performance.py
│   │   ├── my_consolidate_rag.sh
│   │   ├── my_eval_rag.py
│   │   ├── my_eval_rag.sh
│   │   ├── my_finetune_rag_ray.sh
│   │   ├── my_inference_demo.py
│   │   ├── my_retrieval_explore.py
│   │   ├── my_tokenizer_explore.py
│   │   ├── my_use_own_knowledge_dataset.py
│   │   ├── my_use_own_knowledge_dataset.sh
│   │   ├── parse_dpr_relevance_data.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── scratch-cmds.txt
│   │   ├── test_distributed_retriever.py
│   │   ├── _test_finetune_rag.py
│   │   ├── use_own_knowledge_dataset.py
│   │   └── utils_rag.py
│   ├── import_utils.py
│   └── transf_utils.py
└── tutorials
```

Note that `src/hf-finetune-rag` is a modifed version of a training script repo from huggingface-transformers. Within this folder, m ost of the added altered files have the prefix "my_". 

- `models`: root folder containing all fine-tuned and custom pytorch models
- `models/blank`: contains pytorch models with only architectural augmentation
- `models/blank/aug_qencoder`: DPR question encoder with w2v augmented vectors
- `models/blank/w2v_aug_rag`:  RAG model with DPR question encoder integrated into architecture
- `models/full_run`: RAG achitectures finetuned on the CS wikipedia articles dataset
- `data/cs_docs_18k`: Wikipedia articles dataset related to CS
- `data/cs_docs_18k/docs_40_w2v.csv`: 40% of all wikipedia docs whose title keyword also had word2vec entries
- `data/cs_docs_18k/cs-wiki-docs.tsv`: all 18k wikipedia documents related to CS keywords
- `data/faiss_indexes`: faiss indexes of subset of documents used as the knowledge base in RAG 
- `data/training_data`: inp-target rows of training data derived from articles (input is article title + info, output is article outline)
- `data/word2vec_embs`:  Contains python dict pickle files storing word2vec embeddings related to training data and files able to convert between tokenized inputs (both question and context tokenized inputs) to word2vec embeddings
- `src/`: root folder where all custom models and training scripts reside
- `src/custom_qencoder.py`: custom PyTorch question encoder model definition (one with augmented word2vec embeddings + fully connected layer)
- `src/data-prep`: scripts related to parsing Wikipedia articles file, creating and splitting up data into train/test/knowledge base splits, and caching word2vec entries of training data 
- `src/data-prep/store_embeddings.py`: runs Edward's word2vec model on training data and caches embeddings to use later
- `src/data-prep/connect_tok_embs.py`: creates python dicts that can convert between tokenized inputs and corresponding word2vec embedding for cs keywords. Used in pytorch models to add w2v embedding vectors in architecture
- `src/data-prep/create_index_and_split.py`: takes 18k cs documents, splits part of them out to be used in knowledge base, and the rest are split into train/test/val for model training. 
- `src/hf-finetune-rag`: main scripts containing PyTorch model building scripts and training scripts. Modified version of [this original repo](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)
- `src/hf-finetune-rag/my_use_own_knowledge_dataset.sh`: builds out the FAISS index for the CS wiki documents using their DPR and word2vec embeddings
- `src/hf-finetune-rag/my_consolidate_rag.sh`: saves the custom RAG architecture (with word2vec-augmented question encoder) that can be used later for training
- `src/hf-finetune-rag/my_finetune_rag_ray.sh`: script that trains and saves any RAG architecture
- `src/hf-finetune-rag/my_retrieval_explore.py`: see what documents by the retrieval component of any RAG architecture for some query
- `src/hf-finetune-rag/my_inferece_demo.py`: run any full RAG architecture on any input
- `src/hf-finetune-rag/my_eval_rag.sh`: evaluate a fine-tuned RAG on testing data
- `src/hf-finetune-rag/my_calc_performance.py`: calculate precision and recall after running evaluation script on testing data from above


# Usage (Functional Design)

Note: a lot of these steps can be skipped if you are okay with using the default data / model / trainer corresponding to that step. Most of the outputs for each step are already present in the above Google drive zip folders. 

Also, you do not have to follow the exact input / output file names listed in the steps. Just make sure to keep the file names consistent across the steps if you choose different file paths.

The below steps outline what you would need to do for a fully customized training pipeline:

## Prepare data 

1. In `src/data-prep/store_embeddings.py`, set the values of the variables to the following values:
    * `docs_path`: path to `article_jsons.txt`
    * `model_path`: path to `word2vec.model`
    * `embs_out_path`: some path output for output `w2v_embs_dict.pickle`

    and run

    ```bash
    python3 src/data-prep/store_embeddings.py
    ```

2. Run `src/data-prep/connect_tok_embs.py` twice, once with line 13 uncommented and line 14 commeted with 
    * `tok_embs_path`: some path output for `w2v_embs_dict.pickle`

    and once with line 13 commented and line 14 uncommeted with 

    * `tok_embs_path`: some path for output for `ctx_w2v_embs_dict.pickle`

    In both runs, set 
        
    * `embs_dict_path`: path to `w2v_embs_dict.pickle` from before steps

3. Run `src/data-prep/create_index_and_split.py` with
    * `embs_dict_path`: path to `tok2embs_dict.pickle`
    * `articles_file`: path to `article_jsons.txt`
    * `docs_out_file`: some path for output to `docs_40_w2v.csv`
    * `train_data_dir`: some path to a new directory for training data output, e.g. `cs_train_data_w2v`

## Optional: Create a Custom RAG architecture PyTorch module

4. Modify the below files to make any architectural changes to the question encoder, knowledge base index, context encoder, and/or generator. You can also make any other additions / deletions to the RAG architecture. 
    * `src/hf-finetune-rag/my_use_own_knowledge_dataset.py`
    * `src/hf-finetune-rag/my_use_own_knowledge_dataset.sh`
    * `src/custom_qencoder.py`
    * `src/hf-finetune-rag/consolidate_rag_checkpoint.py`
    * `src/hf-finetune-rag/my_consolidate_rag.sh`

    By default, the model with word2vec augmented embeddings for the question encoder and context encoder is used. 

5. In `src/hf-finetune-rag/my_consolidate_rag.sh`, set
    * ` --dest`: some path to the w2v_aug_rag directory (containing custom architecture files)

## Create Knowledge Base FAISS Index and Train

6. Inside the folder `src/hf-finetune-rag`, in `my_use_own_knowledge_dataset.py`, set:

    * `embs_dict_path`: path to `ctx_tok2embs_dict.pickle`

    and in `my_use_own_knowledge_dataset.sh`, set

    * `INP_DOCS_FILE`: path to `docs_40_w2v.csv`
    * `OUT_DIR`: some path output for `cs_docs_index_w2v`

    Then run 

    ```bash
    source src/hf-finetune-rag/my_use_own_knowledge_dataset.sh
    ```

7. In `src/hf-finetune-rag/my_finetune_rag_ray.sh`, set 

    * `DATA_DIR`: path to `cs_train_data_w2v` directory
    * `MODEL_NAME_OR_PATH`: path to `w2v_aug_rag`
    * `OUTPUT_DIR`: some path to `w2v_kb_qenc` (files for the finetuned RAG model)
    
    You may need to also modify `MODEL_TYPE` depending on if you customized your RAG architecture.

    then finetune the model by running

    ```bash
    source src/hf-finetune-rag/finetune_rag_ray.sh
    ```

## Test and Evaluate Finetuned Model [Experimental Scripts]

8. To run the model on a few test inputs, in `src/hf-finetune-rag/my_inference_demo.py`, set
    * `model_root_dir`: path to `w2v_kb_qenc`

    and run
    ```bash
    python my_inference_demo.py
    ```

9. [Experimental] To run the model on all of the test data, run `src/hf-finetune-rag/my_eval_rag.py`. 


10. [Experimental] To calculate metrics like precision and recall of the generated outlines, run `src/hf-finetune-rag/my_calc_performance.py`


# Demo video - In Progress

https://youtu.be/7hJqjkSTey4

# Methodology (Algorithmic Design)

See [Research-report.md](Research-report.md)

# Issues and Future Work

See [Research-report.md](Research-report.md)

