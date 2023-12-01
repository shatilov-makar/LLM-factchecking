#!python3_7\Scripts\python.exe
import spacy
import neuralcoref
import sys
from typing import Any
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def status_gpu_check() -> dict:
    return {
        "message": "I am ALIVE!"
    }


class TextInput(BaseModel):
    input: str
    parameters: dict or None


nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


def coref_resolution(text):
    doc = nlp(text)
    return (doc._.coref_resolved)


@app.post('/coref_resolution')
async def sum_two_digits(data: TextInput) -> dict:
    try:
        text = data.input
        result = coref_resolution(text)
        return {"coref_resolution": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
