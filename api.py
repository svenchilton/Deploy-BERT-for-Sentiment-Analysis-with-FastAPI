from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Dict, List, Union

from model import Model, get_model

import ml_monitor

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


if __name__ == "__main__":
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn

    ngrok_tunnel = ngrok.connect(8000)
    with open('public_url', 'w') as public_url_file:
        public_url_file.write(ngrok_tunnel.public_url)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run('api:app', port=8000)

