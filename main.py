from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conint
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

app = FastAPI()

class Document(BaseModel):
    id: conint(gt=0)
    text: str

class RequestData(BaseModel):
    query: str
    documents: List[Document]

    def construct_pairs(self):
        return [[self.query, doc.text] for doc in self.documents]

class ResponseData(BaseModel):
    id: conint(gt=0)
    similarity: float

@app.post("/api/v1/rerank")
async def rerank_documents(request: RequestData):
    response = []
    pairs = request.construct_pairs()
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True,
                           return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        result = zip(request.documents, scores)
        for doc, score in result:
            response.append({"id": doc.id, "similarity": score.item()})
    response.sort(key=lambda elt: elt['similarity'], reverse=True)
    return {"data": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787)
