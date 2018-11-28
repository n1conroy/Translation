#!/usr/bin/python
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from ppretty import ppretty
import warnings
warnings.filterwarnings("ignore")

threshold = 80

dict_path = "C:\develop\DataScienceMaster\Translate\data\dictionary.tsv"
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
stops= set(stopwords.words('english'))
file_path = "C:\develop\DataScienceMaster\Translate\data\input_text.txt"
model = Word2Vec.load("saved_models/embeddings.model")


class textLabel :
    def __init__(self, label, pos, length, thresh):
        self.label = label
        self.pos = pos
        self.thresh = thresh

    def __eq__(self, other):
        return self.label==other.label and self.pos == other.pos

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__ (self):
         return hash((self.label,self.pos))

    def displayLabel(self):
         print ("The label is " + self.label)

    def displayPosition(self):
         print ("The position is ", self.pos)

    def displayThreshold(self):
         print ("The thresholds used is ", self.thresh)

    def getDefinition():
         print ("The returned definition goes here")


class Match(object):
    def __init__(self, candidate, dict_term, definition):
        self.candidate=candidate
        self.dict_term=dict_term
        self.definition=definition
      

def read_file():
  with open(file_path, 'r') as ff:
    content = ff.read()
  ff.close()
  return(content)

def remove_stopwords(content):
  words = get_words(content)
  filtered_sentence =[]
  for w in words:
    if w not in stops:
        filtered_sentence.append(w) 
  return(filtered_sentence)
         
def read_dictionary_unigrams():
    dict_terms ={}
    with open(dict_path, "r") as f:
         reader = csv.reader(f, delimiter='\t')
         for key,value in reader:
            dict_terms[key]=value;
    f.close()
     
    return (get_dictionary_ngrams(dict_terms,1))

def search_dictionary(targets, d, threshold):
   likely_matches = []
   for target in targets:
     for entry,definition in d.items():
       if (fuzz.ratio(target, entry) > threshold ): 
          match = Match(target, entry, definition)
          likely_matches.append(match)
   return (likely_matches)

def get_sentences(content): 
   return (nltk.sent_tokenize(content))

def get_words(content):
   return (nltk.word_tokenize(content))

def get_indicies(terms, s):
   indicies =[]
   for t in terms:
      i = s.index(t)
      indicies.append(i)
   return (indicies)

def get_dictionary_unigrams():
    dict_terms ={}
    with open(dict_path, "r") as f:
         reader = csv.reader(f, delimiter='\t')
         for key,value in reader:
            dict_terms[key]=value;
    f.close()
    unigrams ={}
    pattern = unigram_pattern
    for key,value in dict_terms.items():
       match= re.match(pattern,key)
       if match:
          unigrams[key]=value
    return (unigrams)

def remove_sentence_word(term, sentence):
    if term in sentence: sentence.remove(term)
    return sentence
    
content = read_file()

for s in get_sentences(content):
  candidates=[]
  first_ver = remove_stopwords(s)
  try:
    candidate1 = model.wv.doesnt_match(first_ver)
    second_ver = remove_sentence_word(candidate1, first_ver)
    candidate2 = model.wv.doesnt_match(second_ver)
    candidates.append(candidate1)
    candidates.append(candidate2)
  except:
    print ("no candidates found in a sentence")
    continue;
  matches = search_dictionary(candidates, get_dictionary_unigrams(), threshold)
  print ("\nLikely matches for the sentence: ",s," With the candidiates ", candidates, " are the matches: ")
  if bool(matches): 
    for m in matches:
        print (ppretty(m))
        #indicies=get_indicies(candidates,s)
        #print ("The index values are")
        #print (indicies)
        #add_label(s, candidate, index, confidence, length=1)
  else:
    print ("nothing resembling that candidate found in dictionary")

'''
  if (match):
  	



textLabels = []
tokenized = get_tokens(content)

for thresh in thresholds:
  token_idx = 0
  for t in (tokenized):
    token_idx=token_idx+1
    for d in read_dictionary():
      if (fuzz.ratio(t,d) > thresh):
         tl1 = textLabel(t, token_idx, thresh)
         textLabels.append (tl1)
         
for k in textLabels:
  k.displayLabel()
  k.displayPosition()
  k.displayThreshold()
  print ('======================================')
'''





	    
