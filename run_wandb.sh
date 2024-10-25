for method in 'BM25_V1' 'BM25_V2' 'BM25_PLUS' 'BM25_PLUS_V2' 'DLITE' 'DLITE_CBRT'
do
    python test_wandb.py $method ${method}_NormalStop
done