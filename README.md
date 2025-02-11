# Deploy BERT for Sentiment Analysis with FastAPI

Deploy a pre-trained BERT model for Sentiment Analysis as a REST API using FastAPI

## Demo

[The model](https://huggingface.co/lvwerra/bert-imdb) is trained to classify sentiment (negative or positive) on the [IMDB 50K movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The webservice is hosted at `localhost`, aka `127.0.0.1`, and at a randomly generated ngrok URL which other users can access. The ngrok address is printed in the command line when generated and saved in the generated file `public_url` for later use. From the command line, you can use the web service to make inferences in the following ways:

1. A single string
```bash
http POST http://127.0.0.1:8000/predict text="Most perceptive observation, Captain"
```
The response will look something like this:
```js
{
    "results": [
        {
            "label": "positive",
            "score": "0.9896888732910156"
        }
    ]
}
```

2. A list of strings
```bash
http POST localhost:8000/predict text:='["Most perceptive observation, Captain", "I aint afraid of no ghost", "The Star Wars prequels have stunning visual effects, but middling storytelling, directing, and acting"]'
```
The response will look something like this:
```js
{
    "results": [
        {
            "label": "positive",
            "score": "0.9896888732910156"
        },
        {
            "label": "positive",
            "score": "0.762516975402832"
        },
        {
            "label": "negative",
            "score": "0.9728572964668274"
        }
    ]
}
```

3. A text file in which each line contains a passage to be analyzed
```bash
http POST $(more public_url)/predict text=@example.txt
```
where `example.txt` contains the lines
```
Most perceptive observation, Captain
I ain't afraid of no ghost! 
The Star Wars prequels have stunning visual effects, but middling storytelling, directing, and acting
```
The response will look something like this: 
```js
{
    "results": [
        {
            "label": "positive",
            "score": "0.9896888732910156"
        },
        {
            "label": "positive",
            "score": "0.9740843772888184"
        },
        {
            "label": "negative",
            "score": "0.9728572964668274"
        }
    ]
}
```
Notice how the model perceives the middle passage as significantly more positive in this example than the prior one, thanks to the addition of an exclamation point. 

You may mix and match the server options with the types of texts entered into the web API.

<!--- 
You can also [read the complete tutorial here](https://www.curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
--->

## Installation

Clone this repo:

```sh
git clone git@github.com:svenchilton/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI.git
cd Deploy-BERT-for-Sentiment-Analysis-with-FastAPI
```

Install the dependencies:

```sh
pipenv install --dev
```

Enter the virtual environment which the previous command created:

```sh
pipenv shell
```

Download the pre-trained model:

```sh
./bin/download_model
```
The script will create a new directory within `Deploy-BERT-for-Sentiment-Analysis-with-FastAPI` named `bert-imdb`, which will contain the relevant model files.

## Test the setup

Start the HTTP server:

```sh
python api.py &
```
Make sure to include the `&` at the end of the line to start the server in the background and allow for the continued use of the terminal.

Send a test request:

```sh
./bin/test_request
```

## Notes

1. If necessary, change the value of `python_version` in `Pipfile` from the default `"3.8"`. The app should work with Python 3.7 and up.
2. As yet, my attempts to dockerize this app (see `Dockerfile`) and add Prometheus/Grafana monitoring with the `ml_monitor` [repo](https://github.com/wiatrak2/ml_monitor) have failed. 
3. There is currently no front end to the web app. 
4. Special thanks to [Leandro von Werra](https://huggingface.co/lvwerra) for help with integrating the bert-imdb model.

## License

MIT
