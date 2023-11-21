import datetime
print('Importing libraries...')
print(datetime.datetime.now())
import shap
import pandas as pd
import pickle
from collections import Counter
from more_itertools import windowed
from trankit import Pipeline
print('Done.')
print(datetime.datetime.now())
print('')

'''
This is a script for calculating stable keywords based on speeches that were correcly and confidently predicted.
'''

def load_shap_values(number):
    '''
    Load shap values.
    '''
    with open(f"/scratch/project_2006385/otto/results/shap_values/shapvalues_model_rerun{number}.sav", 'rb') as f:
        return pickle.load(f)

    
def load_predictions(number):
    '''
    Load model predictions for those speeches that were correctly and confidently predicted.
    '''
    df = pd.read_csv(f"/scratch/project_2006385/otto/results/data_analysis/model_rerun{number}_predictions_highscore_correct.csv")
    return df


def combine_tokens(shap_values, speech_id, party_label):
    '''
    Combine tokens back into full words and return them with their SHAP value. For combined words, the SHAP value is the max value of the tokens.
    '''
    punct = ['.', ',', '!', '?', '-', '...', ':', ';']
    combine = ''
    values = []
    word_values = []
    idx = -1
    for token, next_token in windowed(shap_values.data[speech_id],2):
        idx += 1
        if token.strip() in punct:
            continue
        elif token.endswith(' ') and len(combine) == 0: # if token ends with space and there is nothing to combine, it is a standalone word
            word_values.append((token.strip().lower(), shap_values.values[speech_id][idx][party_label])) # append word-value tuple to return list 
        else:
            combine += token
            values.append(shap_values.values[speech_id][idx][party_label])
            if token.endswith(' '):
                word_values.append((combine.strip().lower(), max(values)))
                combine = ''
                values = []
            elif not token.endswith(' ') and next_token.strip() in punct:
                word_values.append((combine.strip().lower(), max(values)))
                combine = ''
                values = []
            elif idx+1 == len(shap_values.data[speech_id]): # if this is the last item in iterable
                word_values.append((combine.strip().lower(), max(values)))
                
    return word_values



def count_top_words(word_vals, num_top_words):
    '''
    Count the top words in a given model. Change parameter top_words to change how many words are counted.
    '''
    top_words_dict = {}
    for key in word_vals:
        word_scores = {}
        for sublist in word_vals[key]:
            for word, value in sublist:
                if word not in word_scores:
                    word_scores[word] = [value]
                else:
                    word_scores[word].append(value)

        for word in word_scores:
            word_scores[word] = sum(word_scores[word]) / len(word_scores[word])

        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:num_top_words]
        top_words_dict[key] = [word[0] for word in top_words]
    return top_words_dict

def lemmatize(top_words):
    '''
    Lemmatize top words using trankit lemmatizer.
    '''
    lemmas = {}
    for party in top_words:
        lemmas[party] = trankit_pipeline.lemmatize([top_words[party]])
    
    return reformat_lemmas(lemmas)

def reformat_lemmas(lemmas):
    '''
    Reformat the lemmas and drop extra info added by trankit.
    '''
    top_word_dict_lemmatized = {}
    for party in lemmas:
        top_party_words = []
        for token in lemmas[party]['sentences'][0]['tokens']:
            top_party_words.append(token['lemma'])
        top_word_dict_lemmatized[party] = top_party_words
    return top_word_dict_lemmatized
        

def count_keywords(keywords):
    '''
    Count how many times each keywords appears in all models.
    '''
    for party, kw in keywords.items():
        keywords[party] = Counter(keywords[party]).most_common()


def write_kws_to_disk(keywords, num_top_words):
    '''
    Write keywords to file.
    '''
    for party, kws in keywords.items():
        with open(f'../results/data_analysis/shap_keywords/{party}_keywords_combined_50_top{num_top_words}_lemmatized.txt', 'w', encoding='utf-8') as f:
            for tup in kws:
                    f.write(tup[0])
                    f.write(' : ')
                    f.write(str(tup[1]))
                    f.write('\n')

# RUN STARTS HERE!
num_top_words = 1000
keywords = {'SDP': [], 'KOK' : [], 'KESK' : [], 'VIHR' : [], 'VAS' : [], 'PS' : [], 'RKP' : [], 'KD' : []}
party_label_to_name = {0: 'SDP', 1: 'KOK', 2: 'KESK', 3: 'VIHR', 4: 'VAS', 5: 'PS', 6: 'RKP', 7: 'KD'}

# Setup trankit pipeline with Finnish and Swedish for lemmatization
print('Setting up trankit pipeline...')
print(datetime.datetime.now())
trankit_pipeline = Pipeline('finnish')
trankit_pipeline.add('swedish')
trankit_pipeline.set_auto(True) # Auto-detect language
print('Done.')
print(datetime.datetime.now())
print('')

for model_number in list(range(1,51)): # model reruns 1-49
    
    # Load data
    print(f'Getting top {num_top_words} keywords for model {model_number}...')
    print(datetime.datetime.now())
    print('')
    shap_values = load_shap_values(model_number)
    predictions = load_predictions(model_number)
    
    #get dictionary of shap_ids, i.e., the speeches that were given by that party
    speech_ids = {}
    for party in range(8):
        speech_ids[party] = predictions[predictions['label'] == party]['shap_id'].to_list()
    
    # Combine tokens into words and get their respective SHAP values
    party_word_values = {}
    for party_label, speech_id in speech_ids.items():
        word_value_pairs = []
        for idx in speech_id:
            word_value_pairs.append((combine_tokens(shap_values, speech_id = idx, party_label = party_label)))
        party_name = party_label_to_name[party_label]
        party_word_values[party_name] = word_value_pairs

    # Find the most common keywords in this model
    top_words_in_this_model = count_top_words(party_word_values, num_top_words = num_top_words)
    
    # Lemmatize top words
    top_word_lemmas = lemmatize(top_words_in_this_model)
    
    # Add top words to global dictionary
    for key in top_word_lemmas:
        for word in top_word_lemmas[key]:
            keywords[key].append(word)

            
# Get the count of global keywords
count_keywords(keywords)

# Write results to disk
write_kws_to_disk(keywords, num_top_words)
