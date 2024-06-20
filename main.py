"""
This module provides a FastAPI application that uses sequence classification
to rank documents based on their similarity to a given query.

The application accepts POST requests to the '/api/v1/rerank'
endpoint, which takes in a RequestData object containing the query and
a list of Document objects. It then constructs pairs of query and
document texts for scoring. The ranked documents with their
corresponding similarity scores are returned as a ResponseData object.

"""

from uuid import UUID
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

app = FastAPI()

class Document(BaseModel):
    """
    A model representing a document with an ID and text.

    Attributes:
        id Union[int, str, UUID]: The unique ID of the document.
        text (str): The text content of the document.
    """

    id: Union[int, str, UUID]
    text: str

class RequestData(BaseModel):
    """
    A model representing a request to rerank documents based on their
    similarity to a query.

    Attributes:
        query (str): The query string used for comparison.
        documents (List[Document]): A list of Document objects to be
        ranked.

    Methods:
        construct_pairs(): Returns a list of pairs, where each
        pair consists of the query and a document's text.
    """

    query: str
    documents: List[Document]

    def construct_pairs(self):
        """
        Constructs pairs of query and document texts for scoring.

        Returns:
            A list of pairs, where each pair consists of the
            query and a document's text.

        """
        return [[self.query, doc.text] for doc in self.documents]

class ResponseData(BaseModel):
    """
    A model representing the response to a reranking request.

    Attributes:
        id Union[int, str, UUID]: The ID of the ranked document.
        similarity (float): The calculated similarity score between the query
        and the document's text.
    """
    id: Union[int, str, UUID]
    similarity: float

@app.post("/api/v1/rerank")
async def rerank_documents(request: RequestData):
    """
    Ranks a list of documents based on their similarity to a given query using
    sequence classification.

    Args:
        request (RequestData): A RequestData object containing the query and
        a list of Document objects.

    Returns:
        A ResponseData object containing the ranked documents with their
        corresponding similarity scores.
    """

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
