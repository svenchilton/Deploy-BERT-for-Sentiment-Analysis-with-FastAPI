# Deploy BERT for Sentiment Analsysi with FastAPI

Deploy a pre-trained BERT model for Sentiment Analysis as a REST API using FastAPI

## Demo

[The model](https://huggingface.co/lvwerra/bert-imdb) is trained to classify sentiment (negative or positive) on the [IMDB 50K movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). From the command line, you can use the web service to make inferences in the following ways:

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
http POST http://127.0.0.1:8000/predict text:='["Most perceptive observation, Captain", "I aint afraid of no ghost", "The Star Wars prequels have stunning visual effects, but middling storytelling, directing, and acting"]'
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
http POST http://127.0.0.1:8000/predict text=@example.txt
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

Download the pre-trained model:

```sh
./bin/download_model
```

## Test the setup

Start the HTTP server:

```sh
./bin/start_server
```

Send a test request:

```sh
./bin/test_request
```

## License

MIT
