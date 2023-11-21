# This scripts takes the data that is in separate CSV files,
# concatenates them and splits into train, validation and test sets.

from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os

def load_data():
    # Load in data
    all_files = glob.glob("../data/parl_speeches_1980-1989/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files))
    #mask = df['year'].isin(list(range(2010,2020)))
    #df = df[mask]
    return df

def inspect_data(data, name):
    # Let's take a look at the data and how the data are distributed
    print(f'First rows of {name}:')
    print(data.head(10))
    print(f'Label distribution in {name}:')
    print(data['label'].value_counts(normalize=True))
    print(f'Year distribution in {name}:')
    print(data['year'].value_counts(normalize=True))
    print(f'Shape of {name}:')
    print(data.shape)

def replace_id(data):
    data.drop(columns = ['id'], inplace=True)
    ids = []
    for i in range(len(data)):
        ids.append(i)
    data.insert(0, 'id', ids, True)
    return data

def filter(data, num_parties = 3):
    # Labels 0, 1 and 2 (SD, KOK, KESK) are vastly overrepresented

    if num_parties == 3:
    # Let's just keep those three parties to get an even distribution
        print("Dropping smaller parties...")
        bool_mask = data['label'] < 3
        data = data[bool_mask]
    # Let's shuffle the data so it is no longer in chronological order
    data_sample = data.sample(frac=1, random_state = 1234).reset_index(drop=True)
    # Let's print dataset head to see how it looks
    print('Data after shuffling:')
    print(data_sample.head(20))
    return data_sample

def create_splits(data):
    # Split original DataFrame into training and testing sets
    train, test = train_test_split(data, test_size=0.2, random_state=0)

    # Split the test data further in to test and validation sets
    test, validation = train_test_split(test, test_size=0.5, random_state=0)
    
    return train, validation, test

def save_data(datasets, names, num):
    if num == 3:
        num = 'three'
    elif num == 8:
        num = 'eight'
    print('Saving data...')
    for data, name in zip(datasets, names):
        data.to_csv(f'../data/datasets_by_decade/parl_speeches_80s_{num}parties_{name}.csv', index = False)


def main():
    num_parties = 8 #choose either 3 or 8 depending on how many parties you want in the final data
    data = load_data()
    inspect_data(data, 'data')
    data = replace_id(data)
    data = filter(data, num_parties)
    train, validation, test = create_splits(data)
    save_data([train, validation, test], ['train', 'validation', 'test'], num_parties)

if __name__ == '__main__':
    main()
