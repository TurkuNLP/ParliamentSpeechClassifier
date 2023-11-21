""" This is a script for running the classifier.
It is based on https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/hf_trainer_bert.ipynb
Running this script requires a comet api key. Read more: https://www.comet.com/site/
To run from command line, give required args 'xp_name', 'comet_key', 'comet_workspace', and 'comet_project'.
Optional arguments are for changing hyperparameters. Defaults should give decent training results.
"""

# Import comet_ml for logging and plotting
# comet_ml must be imported before any other ML framework for it to work properly
from comet_ml import Experiment
import comet_ml

from pprint import PrettyPrinter
pprint = PrettyPrinter(compact = True).pprint
from datasets import load_dataset
import datasets
import transformers
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import argparse
#import shap
import pickle
import numpy as np


parser = argparse.ArgumentParser(
            description='A script for predicting the political affiliation of the politician who gave a speech in Finnish parliament'
        )
parser.add_argument('--xp_name', required=True,
    help='give a name for this experiment')
parser.add_argument('--comet_key', required=True,
    help='give your comet api key')
parser.add_argument('--comet_workspace', required=True,
    help="give the name of your comet workspace")
parser.add_argument('--comet_project', required=True,
    help="give the name of you comet project")
parser.add_argument('--learning_rate', type=float, default=0.00003,
    help='set trainer learning rate')
parser.add_argument('--batch_size', type=int, default=32,
    help='set trainer batch size')
parser.add_argument('--max_steps', type=int, default=10000,
    help='set trainer max steps')
parser.add_argument('--label_smoothing', type=float, default=0.1,
    help='set label smoothing factor')
args = parser.parse_args()



def comet(key, space, project):
    """
    Create an experiment with your api key
    """
    experiment = Experiment(
    api_key = key,
    workspace = space,
    project_name = project
    )
    return experiment

def get_data():
    """
    Load data from desired csv files.
    """
    cache_dir = '../hf_cache' # hf cache can get bloated with multiple runs so save to disk with enough storage
    #output_dir = f'../results/models/{args.xp_name}' # Where results are saved
    print('Loading dataset...')
    dataset = load_dataset('csv', data_files = {'train': '../data/final_data/parl_speeches_2000-2021_train.csv',
                                                'test': '../data/final_data/parl_speeches_2000-2021_test.csv'},
                                                cache_dir = cache_dir)
    
    dataset=dataset.shuffle()
    return dataset

# Get the number of labels, which is required for the model
def get_labels(dataset):
    """
    Get the labels from data.
    Most larger dataset have all 8 labels, but some might purposefully not have all.
    Dataset of older speeches (before 1980s) also do not have all labels.
    """
    train_labels = dataset['train']['label']
    label_ints = sorted(list(set(train_labels)))
    num_labels = len(set(train_labels))
    id2label = {0: 'SDP', 1: 'KOK', 2: 'KESK', 3: 'VIHR', 4: 'VAS', 5: 'PS', 6: 'RKP', 7: 'KD'}
    id2label_in_data = {}
    for key, value in id2label.items():
        if key in label_ints:
            id2label_in_data[key] = value
    label2id_in_data = {}
    for key, value in id2label_in_data.items():
        label2id_in_data[value] = key
    return num_labels, id2label_in_data, label2id_in_data

def filter_short(dataset):
    """
    Remove speeches that are 12 tokens long or shorter.
    """
    dataset = dataset.filter(lambda x: len(x['input_ids']) > 12)
    return dataset

def get_trainer_args(xp_name, learning_rate, batch_size, max_steps, label_smoothing):
    """
    Get arguments for model trainer.
    """
    trainer_args = transformers.TrainingArguments(
        output_dir = f'../results/models/{xp_name}',
        save_total_limit = 1, #only keep the best model in the end
        evaluation_strategy = 'steps',
        logging_strategy = 'steps',
        load_best_model_at_end = True,
        eval_steps = 100,
        logging_steps = 100,
        save_steps = 100,
        metric_for_best_model = 'eval_macro-f1',
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        max_steps = max_steps,
        label_smoothing_factor = label_smoothing,
    )
    return trainer_args
    
def get_trainer(model, trainer_args, dataset, compute_metrics, data_collator, tokenizer, early_stopping):
    os.environ['COMET_LOG_ASSETS'] = 'True'
    trainer = None
    trainer = transformers.Trainer(
        model = model,
        args = trainer_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        compute_metrics = compute_metrics,
        data_collator = data_collator,
        tokenizer = tokenizer,
        callbacks =[early_stopping]
    )
    return trainer

def predict(name, trainer, dataset):
    """
    Evaluate performance and save metrics to disk.
    """
    pred_results = trainer.predict(dataset["eval"])
    with open(f'../results/models/evaluation_{name}.txt', 'w') as f:
        f.write('Accuracy: ')
        f.write(f'{pred_results[2]["test_accuracy"]}\n')
        f.write('Macro-f1: ')
        f.write(f'{pred_results[2]["test_macro-f1"]}\n')
        f.write('Loss: ')
        f.write(f'{pred_results[2]["test_loss"]}\n')
    
    winning_label = [i.argmax() for i in pred_results.predictions]
    
    def softmax(vec):
        exponential = np.exp(vec)
        probabilities = exponential / np.sum(exponential)
        return probabilities
    probs = [softmax(i) for i in pred_results.predictions]
    winning_prob = [max(i) for i in probs]
    return winning_label, winning_prob

def calculate_shap(data, model, tok):
    """
    Calculate shap values for model explainability.
    """
    pipeline_all_scores = transformers.TextClassificationPipeline(model = model,
                                                                  device = 0,
                                                                  batch_size = 128,
                                                                  tokenizer = tok,
                                                                  truncation = True,
                                                                  padding = True,
                                                                  max_length = 512,
                                                                  return_all_scores=True, #returns all scores, not just winning label
                                                                  )
    explainer = shap.Explainer(pipeline_all_scores, seed=1234, output_names= ['SDP', 'KOK', 'KESK', 'VIHR', 'VAS', 'PS', 'RKP', 'KD'])
    shap_values = explainer(data['text'])
    return explainer, shap_values
    
def save_shap(explainer, shap_values, xp_name):
    """
    Save explainer and shap values.
    """
    filename_expl = f'../results/explainer_{xp_name}.pickle'
    with open(filename_expl,"wb") as f:
        pickle.dump(explainer, f)

    filename_values = f'../results/shapvalues_{xp_name}.pickle'
    with open(filename_values, 'wb') as f:
        pickle.dump(shap_values, f)
        
def keep_true_highscores(data, winning_label, winning_prob, name):
    df = data['test'].to_pandas()
    df['predicted_label'] = winning_label
    df['prediction_score'] = winning_prob
    prob_mask = df['prediction_score'] > 0.9
    correct_mask = df['predicted_label'] == df['label']
    df = df[prob_mask]
    df = df[correct_mask]
    df = df.reset_index(drop = True)
    df.to_csv(f'../results/data_analysis/{name}_predictions_highscore_correct.csv', index_label = 'shap_id')
    return df
    
def main(args):
    
    def tokenize(example):
        """
        Tokenize data.
        """
        return tokenizer(
            example['text'],
            max_length=512,
            truncation=True
        )
  
    def get_example(index):
        """
        Get example for calculating accuracy and f1-score.
        """
        return dataset['test'][index]['text']

    def compute_metrics(pred):
        """
        Function for computing evalution metrics.
        """
        experiment = comet_ml.get_global_experiment()

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average = 'macro'
        )
        acc = accuracy_score(labels, preds)

        if experiment:
            step = int(experiment.curr_step) if experiment.curr_step is not None else 0
            experiment.set_step(step)
            experiment.log_confusion_matrix(
                y_true = labels,
                y_predicted = preds,
                file_name = f'confusion-matrix-step-{step}.json',
                labels = list(id2label_in_data.values()),
                index_to_example_function = get_example,
            )

        return {'accuracy': acc, 'macro-f1': f1, 'precision': precision, 'recall': recall}
    
    # Run starts here.
    # Start logging.
    experiment = comet(args.comet_key, args.comet_workspace, args.comet_project)
    
    # Get data.
    dataset = get_data()
    num_labels, id2label_in_data, label2id_in_data = get_labels(dataset)
    
    # Setup model and tokenizer
    model_name = 'TurkuNLP/bert-base-finnish-cased-v1'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                            num_labels = num_labels, 
                                                                            id2label = id2label_in_data,
                                                                            label2id = label2id_in_data)
    trainer_args = get_trainer_args(args.xp_name, args.learning_rate, args.batch_size, args.max_steps, args.label_smoothing)
    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience = 10)
    
    # Tokenize and remove short speeches
    dataset = dataset.map(tokenize, batched=True)
    dataset = filter_short(dataset)

    # Train model
    trainer = get_trainer(model, trainer_args, dataset, compute_metrics, data_collator, tokenizer, early_stopping)
    trainer.train()
    
    # Get predictions from and evaluate model
    pred_label, pred_prob = predict(args.xp_name, trainer, dataset)
    
    # Keep only speeches that were correctly predicted with > 90% prediction probability
    #shap_data = keep_true_highscores(dataset, pred_label, pred_prob, args.xp_name)
    
    # Get shap values
    # These cause OOM errors, so I'm skipping them and calculate SHAP values separately
    #shap_explainer, shap_values = calculate_shap(shap_data, model, tokenizer)
    #save_shap(shap_explainer, shap_values, args.xp_name)

if __name__ == '__main__':
    main(args)
