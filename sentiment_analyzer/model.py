import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = 0 if torch.cuda.is_available() else -1

        # Model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["BERT_MODEL"])

        # Model configuration
        auto_config = AutoConfig.from_pretrained(config["BERT_MODEL"])
        auto_config.id2label = {i: name for i, name in enumerate(config["CLASS_NAMES"])}
        auto_config.label2id = {name: i for i, name in enumerate(config["CLASS_NAMES"])}

        # Model weights
        model = AutoModelForSequenceClassification.from_pretrained(config["BERT_MODEL"], config=auto_config)

        # Build the pipeline
        self.classifier = pipeline('sentiment-analysis', model=model, tokenizer=self.tokenizer, 
                                   device=self.device, return_all_scores=False)

        #self.classifier = classifier.to(self.device)

    def predict(self, text):
        return self.classifier(text)


# model = Model()


def get_model():
    #return model
    return Model()
