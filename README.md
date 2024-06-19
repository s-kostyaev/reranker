# reranker
Reranker local service. Can be useful as a part of RAG pipeline.

You can run application in docker with:

``` shell
docker compose up
```

And call service with:

``` shell
curl -X POST "http://127.0.0.1:8787/api/v1/rerank" -H "Content-Type: application/json" -d '{"query":"what is panda?", "documents": [{"id": 1, "text": "hi"}, {"id": 2, "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."}, {"id": 3, "text": "I like pandas."}]}'
```

You will recieve response contains id and similarity fields. It will
be sorted by similarity in descending order:

``` json
{"data":[{"id":2,"similarity":5.265044212341309},{"id":3,"similarity":-7.278249263763428},{"id":1,"similarity":-8.183815002441406}]}
```
