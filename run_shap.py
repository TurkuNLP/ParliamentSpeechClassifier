import transformers
import numpy as np
import torch
import shap
import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True)
parser.add_argument('--data_file', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

def setup(path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

def predict(df):
    """
    Get predictions
    """
    prediction_pipeline = transformers.TextClassificationPipeline(model = model,
                                                                  device = 0,
                                                                  tokenizer = tokenizer,
                                                                  truncation = True,
                                                                  padding = True,
                                                                  max_length = 512,
                                                                  return_all_scores=False, #return only winning label
                                                                  )
    winning_labels = []
    winning_scores = []
    l2id = {'SD': 0, 'SDP': 0, 'KOK': 1, 'KESK': 2, 'VIHR': 3, 'VAS': 4, 'PS': 5, 'R': 6, 'RKP': 6, 'KD': 7}
    for speech in df['text']:
        winning_labels.append(l2id[prediction_pipeline(speech)[0]['label']])
        winning_scores.append(prediction_pipeline(speech)[0]['score'])
    df['predicted_label'] = winning_labels
    df['prediction_score'] = winning_scores

    return df

def keep_true_highscores(df, name):
    """
    Keep only those speeches which are confidently are correctly predicted.
    """
    prob_mask = df['prediction_score'] > 0.9
    correct_mask = df['predicted_label'] == df['label']
    df = df[prob_mask]
    df = df[correct_mask]
    df = df.reset_index(drop = True)
    df.to_csv(f'../results/data_analysis/{name}_predictions_highscore_correct.csv', index_label = 'shap_id')
    return df


def setup_pipeline(model, tokenizer):
    pipeline = transformers.TextClassificationPipeline(model = model,
                                                       device = 0,
                                                       batch_size = 128,
                                                       tokenizer = tokenizer,
                                                       truncation = True,
                                                       padding = True,
                                                       max_length = 512,
                                                       return_all_scores=True, #returns all scores, not just winning label
                                                       )
    return pipeline

def setup_explainer(pipeline):
    explainer = shap.Explainer(pipeline, seed=2345, output_names= ['SDP', 'KOK', 'KESK', 'VIHR', 'VAS', 'PS', 'RKP', 'KD'])
    return explainer

def calculate_shap(explainer, dataset):
    shap = explainer(dataset['text'])
    return shap


def save_shap(explainer, shap, model_name):
    # Save explainer and shap values.
    with open(f'../results/shap_values/explainer_{model_name}.sav',"wb") as f:
        pickle.dump(explainer, f)

    with open(f'../results/shap_values/shapvalues_{model_name}.sav',"ab") as f:
        pickle.dump(shap, f)

print(f'Calculating SHAP values for data in {args.data_file} for model in checkpoint {args.model}.')
model_name = args.model_name
model, tokenizer = setup(load_dir = args.model)
data_name = args.data_file
dataset = load_data(data_name)
dataset = predict(dataset)
dataset = keep_true_highscores(dataset, model_name)
pipeline = setup_pipeline(model, tokenizer)
explainer = setup_explainer(pipeline)
shap = calculate_shap(explainer, dataset)
save_shap(explainer, shap, model_name)
