# Introduction

In this work, we tackle the task of outline generation for articles using a modified version of the Retrieval Augmented Generation framework. Specifically, given some knowledgebase of existing Wikipedia documents and a text input of the form

`<topic>: <description about topic>`

We aim to generate a list of possible subheadings for an article about this topic (e.g. ['Applications', 'History', 'Types']). 




# Architecture

## Baseline

As a baseline for this task, we train a T5 model to learn the sequence-to-sequence task of directly mapping from the text input to the section headings. 

## Modifications

We chose to use a modified version of the RAG (Retrieval Augmented Generation) architecture. The main intuition behind RAG is to augment an input sequence with relevant context text from some knowledge base of documents and then train a generator to convert it from one input sequence to another. 

![Orig RAG overview](documentation-pics/RAG-overview.png)

As shown in the above picture, the main components of the RAG architecture are the retreiver and the generator. Our main modifications are centered around the retriever. Specifically, we augment `q(x)` by concatenating another feature vector representative of the input. We also augment `d(z)` with the same kind of feature vector. We add an additional fully connected layer on `q(x)` after augmenting it to allow flexiblity in the dimensions for the query embedding and document / context embedding spaces. 

Mathematically,
* $q'(x) = W \times (q(x) \bigoplus v(x))$
* $d'(z) = d(z) \bigoplus v(z)$

where $W$ is a learnable parameter, $v(x)$ is some additional features of $x$, and $\bigoplus$ is the vector concatenation operator. Note that $v(x)$ doesn't necessarily solely rely on the text of the article $x$. It could rely on additional meta data about the article (e.g. graphical embedding of the article). 

Additionally, unlike in the original RAG architecture, the text that is used to generate the representation from $q(x)$ is different from the text that is passed down to the generator component.

![V1 Extended RAG](documentation-pics/outline-gen-framework-1.jpg)

By formatting our architecture this way, we are forcing the Retreiver part of our architecture to learn to fetch similar / relevant documents, and the generator to create a summarized outline from the outlines of several similar documents. In a sense, this is like a learnable kNN model: the retriever learns to fetch similar documents based on some query embedding in a similar space (like fetching nearest neighbors), and the generator takes the outlines from these neighbors to generate the output (generating the output from the nearest-neighbors part). 


# Data Collection
Training data was scraped from a subset of wikipedia articles related to Computer Science. In practice, the training articles can be on arbitrary topics. The initial data was obtained from by obtaining a list of Computer Science keywords (mined from places such as arXiV and Springer), and checking whether there was a Wikipedia article corresponding to that keyword. 

Out of around 18,000 computer science wikipedia articles, 40% were used to populate the knowledge base. The reamining 60% were split in a 80-18-2 ratio for train-test-val data. 

# Pending work / Known Issues

* Learning rate / decoding method for last trained model produces repetitve results.

* Need to test on additional feature representations like word2vec. 

* There was a second variant for extending the RAG that was never implemented / tested out. The main difference between this variant and the first variant was to pass in the word2vec embedding as the first token in the embedding layer of the BERT architecture.

![V2 Extended RAG](documentation-pics/outline-gen-framework-2.jpg)


# References

* Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., & Kiela, D.. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.