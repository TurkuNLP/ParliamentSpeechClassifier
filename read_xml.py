# Code for parsing and converting semparl XML files to CSV.
# This works for files between 2000 to 2021,
# parsing older files requires some modification.
# Output is a CSV file with columns id, text, label, speaker, year

# import modules
import csv
import xml.etree.ElementTree as et
import pandas as pd
import argparse
from more_itertools import windowed


parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, required=True,
    help='which year of speeches to parse')
parser.add_argument('--appendix')
args = parser.parse_args()

#initialise empty lists for later
speaker_name = []
speaker_party = []
speech_text = []
speech_year = []

# The year you want to parse from xml to csv
if args.appendix:
   year = str(args.year)+args.appendix
else:
   year = int(args.year)


# Parsing the XML file
if isinstance(year, int):
   if 2015 <= year <= 2021:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/2015-2021/Speeches_{year}.xml')
   elif 2000 <= year < 2015:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/2000-2014/Speeches_{year}.xml')
   elif 1990 <= year < 2000:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1990s/Speeches_{year}.xml')
   elif 1980 <= year < 1990:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1980s/Speeches_{year}.xml')
   elif 1970 <= year < 1980:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1970s/Speeches_{year}.xml')
   elif 1960 <= year < 1970:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1960s/Speeches_{year}.xml')
   elif 1950 <= year < 1960:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1950s/Speeches_{year}.xml')
   elif 1940 <= year < 1950:
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1940s/Speeches_{year}.xml')
   else:
      raise IndexError('Not a valid year')
else:
   if year == '1975_II':
      xmlparse = et.parse(f'../data/semparl-speeches-data-master/XML/1970s/Speeches_{year}.xml')

#get the root element
root = xmlparse.getroot()

#initialise empty dictionary for later
party_dict = {}

#get surname, forename and party of participants in document order
for person in root.findall('./teiheader/profileDesc/particDesc/listPerson/person'):
   name = person.attrib['{http://www.w3.org/XML/1998/namespace}id'] #get person name in format firstname_lastname
   try:
      party = person.find('affliation').attrib['ref'] #party is given as attribute of affliation element
      party = party.replace('#party.','') #each party name is preceded by '#party." e.g. '#party.SDP' which we want to remove
      party_dict.update({name : party}) #add key-value pair to dictionary for later retrieval
   except:
      party = 'N/A' #some speakers miss party information, so in that case they are given party label N/A
      party_dict.update({name : party})

speech_segments = windowed(root.findall('./TEI/text/body/div/div/u'), 2)
interrupted = ''
for segment, next_segment in list(speech_segments): #each speech segment is in element 'u'
   text_segment = segment.text #get the text of the speech
   if not isinstance(text_segment,str): #check that the text is actually str, there are some 'nan's in the data
      continue
   if 'ana' in segment.attrib: #if speaker has attribute 'ana', they are chairman, vicechair or secondvicechair. we want to remove them from the data
      continue
   if 'next' in segment.attrib or 'prev' in segment.attrib: #segments with next and prev are interrupted
      interrupted += text_segment #interrupted segments are combined to form one speech
      if 'next' in segment.attrib and 'prev' not in next_segment.attrib: # check if there are segments with 'next' not followed by 'prev'
         print('Error', text_segment) #if there are, locate problem
         continue
      elif 'next' in segment.attrib: # if segment only has next, speech continues in next segment
         continue
      elif 'prev' in segment.attrib and 'prev' in next_segment.attrib: # sometimes segments with only 'prev' still continue in next segment
         continue
      elif 'prev' in segment.attrib and 'prev' not in next_segment.attrib: #if segment has only attribute 'prev' and next one doesn't, it's the end of the speech
         text_segment = interrupted
         interrupted = '' 
   speaker = segment.attrib['who'] #get the name of the speaker
   speaker = speaker.replace('#','') #remove preceding hashtag from speaker name
   speaker_name.append(speaker) #append name of speaker to a list
   speaker_party.append(party_dict[speaker]) #get party of speaker from dictionary and append to separate list
   speech_text.append(text_segment) #append the text of the speech to a third list
   speech_year.append(year)


#create pandas dataframe where each row contains speech, party label, name of speaker and year
df = pd.DataFrame({'text': speech_text, 'label': speaker_party, 'speaker': speaker_name, 'year': speech_year})

df.loc[df['label'] == 'SDP', 'label'] = 'SD' #change few occurrances of SDP to more commonly used SD
df.loc[df['label'] == 'RKP', 'label'] = 'R' #change few occurrances of RKP to more commonly used R
df.loc[df['label'] == 'SKL', 'label'] = 'KD' #change old name SKL to the new name of the party KD
df.loc[df['label'] == 'SMP', 'label'] = 'PS' #change old name SMP to the new name of the party PS
df.loc[df['label'] == 'SKDL', 'label'] = 'VAS' #change old name SKDL to the new name of the party VAS

#mapping for converting party names to label numbers, because transformer requires numbered labels
if isinstance(year, int):
   if year >= 1983: 
      label_dict = {'SD': 0, 'KOK': 1, 'KESK': 2, 'VIHR': 3, 'VAS': 4, 'PS': 5, 'R': 6, 'KD': 7}
   else:
      label_dict = {'SD': 0, 'KOK': 1, 'KESK': 2}
else:
   label_dict = {'SD': 0, 'KOK': 1, 'KESK': 2}

#do the conversion
for key in label_dict.keys():
    df.loc[df['label'] == key, 'label'] = label_dict[key]
    
values = list(label_dict.values()) #get list of valid label values

#print labels before dropping to check if there is something interesting there
ls = df['label'].to_list()
print('labels in data before dropping invalid values:')
print((set(ls)))

df = df[df["label"].isin(values)] #drop rows with invalid label values, i.e. parties that do not apper in label_dict

#print all labels to check that there are no errors
ls = df['label'].to_list()
print('labels in clean data:')
print(set(ls))

# After dropping some speeches, we want to reset index so it matches line number again
df = df.reset_index(drop=True)

#save dataframe as csv for later use
if isinstance(year, int):
   if 2015 <= year <= 2021:
      df.to_csv(f'../data/parl_speeches_2000-2021/parl_speeches_{year}.csv', index_label = 'id')
   elif 2000 <= year < 2015:
      df.to_csv(f'../data/parl_speeches_2000-2021/parl_speeches_{year}.csv', index_label = 'id')
   elif 1990 <= year < 2000:
      df.to_csv(f'../data/parl_speeches_1990-1999/parl_speeches_{year}.csv', index_label = 'id')
   elif 1980 <= year < 1990:
      df.to_csv(f'../data/parl_speeches_1980-1989/parl_speeches_{year}.csv', index_label = 'id')
   elif 1970 <= year < 1980:
      df.to_csv(f'../data/parl_speeches_1970-1979/parl_speeches_{year}.csv', index_label = 'id')
   elif 1960 <= year < 1970:
      df.to_csv(f'../data/parl_speeches_1960-1969/parl_speeches_{year}.csv', index_label = 'id')
   elif 1950 <= year < 1960:
      df.to_csv(f'../data/parl_speeches_1950-1959/parl_speeches_{year}.csv', index_label = 'id')
   elif 1940 <= year < 1950:
      df.to_csv(f'../data/parl_speeches_1940-1949/parl_speeches_{year}.csv', index_label = 'id')
else:
   if year == '1975_II':
      df.to_csv(f'../data/parl_speeches_1970-1979/parl_speeches_{year}.csv', index_label = 'id')

