# Unifying Non-Semantic Approaches for Multilingual Document Retrieval


## Viewing the code
The main chunk of the code is present in `src/` and is well commented and documented. The files in the main directory are to run the code, and explained below:

## Running the code

1. Install relevant libraries
```
pip install -r requirements.txt
```

2. To reproduce the results of Kaggle:
```
python test_best.py
```
Running the above also generates the same files as the TFIDF-Dump dataset folder added to Kaggle.

3. To run the code for any of the implementations:
```
python main.py --method [METHOD] [--save_index] --k1 [K1] --b [B] --d [D]


usage: main.py [-h]
               [--method {TFIDF,BM25_V1,BM25_V2,BM25_PLUS,BM25_PLUS_V2,DLITE,DLITE_CBRT,TF_LDP}]
               [--save-index] [--k1 K1] [--b B] [--d D]

options:
  -h, --help            show this help message and exit
  --method {TFIDF,BM25_V1,BM25_V2,BM25_PLUS,BM25_PLUS_V2,DLITE,DLITE_CBRT,TF_LDP}
                        args.method to use for retrieval (default: TFIDF)
  --save-index          Save the index and vocab
  --k1 K1               k1 parameter (default: 1.5)
  --b B                 b parameter (default: 0.75)
  --d D                 delta parameter (default: 1)
```

4. To perform hyperparameter search on any method (on `wandb`):
```
python hyperparam_search.py <method_name>
```

5. To generate the plot in the report, refer to `plots.ipynb`

## Team_EIJ
```
Eeshaan Jain
Imane 
Jeanne
```