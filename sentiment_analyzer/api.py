from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Dict, List, Union

from .classifier.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: Union[str, List[str]]


class SentimentResponse(BaseModel):
    results: List[Dict[str, Union[str, float]]]


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    # When passing the path to a file into a field of an httpie request 
    # with the =@ method, httpie reads in the contents of that file as a 
    # single string. In that case, we'll convert that into a list of 
    # strings by splitting on the newline character \n.
    text = request.text
    # Note that this will also convert single strings without \n to 
    # single-element lists of strings
    if type(text) is not list:
        text = text.split('\n')
    results = model.predict(text)
    return SentimentResponse(results=results)
