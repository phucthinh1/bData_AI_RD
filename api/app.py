from fastapi import FastAPI
from pydantic import BaseModel
from inference.generate import generate

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/chat")
def chat(query: Query):
    response = generate(query.query)
    return {"response": response}
