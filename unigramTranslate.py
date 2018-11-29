#!/usr/bin/python
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from ppretty import ppretty


import warnings
warnings.filterwarnings("ignore")

threshold = 80

#dict_path = "C:\develop\DataScienceMaster\Translate\data\dictionary.tsv"
dict_path = "C:\develop\DataScienceMaster\Translate\data\gangslang.tsv"
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
stops= set(stopwords.words('english'))
file_path = "C:\develop\DataScienceMaster\Translate\data\input_text.txt"

model1 = Word2Vec.load("saved_models/embeddings4.model")
model2 = gensim.models.KeyedVectors.load_word2vec_format('saved_models\GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
#model2 = Word2Vec.load("saved_models/embeddings5.model")

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

def get_index(term, s):
   return (s.index(term))

def get_dictionary_unigrams():
    dict_terms ={}
    with open(dict_path, "r", encoding="utf-8") as f:
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
    

def get_three_predictions(model, sentence):
      cand1 = model.wv.doesnt_match(sentence)
      sentence_v2 = remove_sentence_word(cand1,sentence)
      cand2 = model.wv.doesnt_match(sentence_v2)
      sentence_v3 = remove_sentence_word(cand2,sentence)
      cand3 = model.wv.doesnt_match(sentence_v3)
      return (cand1, cand2, cand3)


content = read_file()


for s in get_sentences(content):
  m1_candidates=[]
  m2_candidates=[]
  sent = remove_stopwords(s)
  try:
      pred1 = get_three_predictions(model1, sent)
      pred2 = get_three_predictions(model2, sent)
      common =list(set(pred1).intersection(pred2))
      print (s)
      print (common)
  except:
      print ("no candidates found in a sentence")
      continue;
  if (common):
      matches = search_dictionary(common, get_dictionary_unigrams(), threshold)
  else:
      print ("NO SLANG: The sentence [", s, "] had no predictions.")
      continue;
  if bool(matches): 
    for m in matches:
        indicies=get_index(m.candidate,s)
        print ("SLANG: The sentence [",s,"] has the possible slang candidiate ", m.candidate, ". It matched with ", m.dict_term)
        print ("its index values is", indicies )
        #add_label(s, candidate, index, confidence, length=1)
  else:
     print ("nothing resembling that candidate found in dictionary")








	   
