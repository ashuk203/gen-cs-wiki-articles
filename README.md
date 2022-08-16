This module predicts the outline for an article given its title and a short sentence description about it. The below part contains details about how to run and setup the code. For a research report, please refer to [Research-report.md](Research-report.md)

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

2. Download and expand the zipped `data` folder in the root folder of this repo from this [Google drive link](https://drive.google.com/file/d/1JM2wofuZJA-Z9d7h68DbYQAVx7SaB-jo/view?usp=sharing)

3. Similarly, download and expand the zipped `models` folder: [Google drive link](placeholder)

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


# Usage (Functional Design) - In Progress

Describe all functions / classes that will be available to users of your module. This section should be oriented towards users who want to _apply_ your module! This means that you should **not** include internal functions that won't be useful to the user in this section. You can think of this section as a documentation for the functions of your package. Be sure to also include a short description of what task each function is responsible for if it is not apparent. You only need to provide the outline of what your function will input and output. You do not need to write the pseudo code of the body of the functions.

- Takes as input a list of strings, each representing a document and outputs confidence scores for each possible class / field in a dictionary

```python
    def classify_docs(docs: list[str]):
        ...
        return [
            { 'cs': cs_score, 'math': math_score, ..., 'chemistry': chemistry_score },
            ...
        ]
```

- Outputs the weights as a numpy array of shape `(num_classes, num_features)` of the trained neural network

```python
    def get_nn_weights():
        ...
        return W
```

# Demo video - In Progress

Make sure to include a video showing your module in action and how to use it in this section. Github Pages doesn't support this so I am unable to do this here. However, this can be done in your README.md files of your own repo. Follow instructions [here](https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md) of the accepted answer

# Methodology (Algorithmic Design)

See [Research-report.md](Research-report.md)

# Issues and Future Work

In this section, please list all know issues, limitations, and possible areas for future improvement. For example:

- High false negative rate for document classier.
- Over 10 min run time for one page text.
- Replace linear text search with a more efficient text indexing library (such as whoosh)
- Include an extra label of "no class" if all confidence scores low.
