# reranker
Reranker local service. Can be useful as a part of RAG pipeline. It
uses
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
under the hood. It is lightweight reranker model with strong
multilingual capabilities.

# How to use

## Docker

You can run application in docker with:

``` shell
docker compose up --build
```

Check available options in `compose.yaml`.

## Local setup

You can run it locally with python:

``` shell
pip install -r requirements.txt
DEVICE=mps MAX_LENGTH=1024 python main.py
```

## Environment variables

- `PORT` - change the port the service listens on. Default 8787.
- `MAX_LENGTH` - maximum sequence length. Default 512 tokens.
- `MODEL` - reranking model. Default 'BAAI/bge-reranker-v2-m3'.
- `DEVICE` - set to `mps` for M-series macs or `cuda` for nvidia
  cards. Cpu will be used if not set.

You can call service with:

``` shell
curl -X POST "http://127.0.0.1:8787/api/v1/rerank" -H "Content-Type: application/json" -d '{"query":"what is panda?", "documents": [{"id": 1, "text": "hi"}, {"id": 2, "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."}, {"id": 3, "text": "I like pandas."}]}'
```

You will recieve response contains id and similarity fields. It will
be sorted by similarity in descending order:

``` json
{"data":[{"id":2,"similarity":5.265044212341309},{"id":3,"similarity":-7.278249263763428},{"id":1,"similarity":-8.183815002441406}]}
```
