# Unifying Non-Semantic Approaches for Multilingual Document Retrieval


## Running the code

1. Install relevant libraries
```
pip install -r requirements.txt
```

2. To run the code for any of the implementations:
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


## Directory Structure

```
.
├── src/
│   ├── custom_stopwords.py
│   ├── methods.py
│   ├── text.py
│   └── utils.py
├── hyperparam_search.py
├── main.py
├── README.md
└── requirements.txt
```

## Team
```
Eeshaan Jain
Imane 
Jeanne
```