# maxilloPRO
specialized maxillofacial prosthetic large language model for education and clinical-decision support

## Setting up the Environment
Tested on miniconda/25.3.1 and Python 3.10.20

Run
```
conda env create -f environment.yml
```

## Features
This repository consist of codes used to create a Naive RAG knowledge base alongside with a sample code on creating an LLM-RAG system.

### run_build.py
Create the base knowledge base from pdf files. 

Current version assumes a 2-column pdf format.

### run_append.py
If you wish to extend the knowledge base with new sources or manually curated ones, you can use this code to do so.

### run_query.py
A sample code on how to extract relevant context based on cosine similarity between User query and all documents.

### run_rag_llm.py
A sample code on how to couple the RAG's output as the input for the LLM.