from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Dict, List, Union

# Import from model.py
from model import Model, get_model

# If a local copy of the ml_monitor repo exists 
# (available at https://github.com/wiatrak2/ml_monitor), 
# import it
import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
ml_monitor_dir = os.path.join(parent_dir, 'ml_monitor')
# ml_monitor_dir = os.path.join(current_dir, 'ml_monitor')
if os.path.exists(ml_monitor_dir):
    # Ensure that the ml_monitor module is set up
    os.chdir(ml_monitor_dir)
    import subprocess
    subprocess.run('pip install .', shell=True)
    os.chdir(current_dir)
    import ml_monitor
    my_monitor = ml_monitor.Monitor()
    my_monitor.start()


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
    # Once server stops running, stop monitoring as well
    print(f'Stopped app server {ngrok_tunnel.public_url}')
    if os.path.exists(ml_monitor_dir):
        print(f'Now stopping Prometheus and Grafana ML monitoring')
        my_monitor.stop()

