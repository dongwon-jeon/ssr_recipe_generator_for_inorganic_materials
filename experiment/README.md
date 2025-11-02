# Prediction for the example models
```bash
# RAG
top_k=5
python predict.py --use_rag --top_k $top_k --batch_size 200 --model o3-mini
python judge.py data/test_high_impact/o3-mini/rag_0211__k${top_k}.jsonl --batch_size 200
```

