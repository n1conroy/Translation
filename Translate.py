#!/usr/bin/python
import csv
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from ppretty import ppretty
#import enchant

import warnings
warnings.filterwarnings("ignore")


ngram_thresholds =[75, 85]

dict_path = "dictionary.tsv"
#dict_path = "C:\develop\DataScienceMaster\Translate\data\gangslang.tsv"
#englishDictionary = enchant.Dict("en_US")

punc_pattern = r'[\S\n\t\r]+'
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
stops= set(stopwords.words('english'))

file_path = "input_text.txt"


#model1 = Word2Vec.load("saved_models/embeddings.model")
#model2 = gensim.models.KeyedVectors.load_word2vec_format('saved_models\embeddingmodel6.bin', binary=True, limit=500000)
#model2 = Word2Vec.load("saved_models/embeddings.model")

class phraseLabel :
    def __init__(self, label, pos, length, defin, found, thresh):
        self.label = label
        self.pos = pos
        self.thresh = thresh
        self.length = length
        self.defin = defin
        self.found = found
    
    def __eq__(self, other):
        return (self.label == other.label and self.pos == other.pos and self.thresh == other.thresh)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__ (self):
         return hash((self.label,self.pos))

class Match(object):
    def __init__(self, candidate, dict_term, definition, threshold):
        self.candidate=candidate
        self.dict_term=dict_term
        self.definition=definition
        self.thresh = threshold

def get_threshold (term, l):
    threshold=80
    if l ==1 and len(term) <5:
        threshold =89
    elif l == 1 and len(term) <6 :
        threshold=88
    elif l ==1 and len(term) <7:
        threshold=84
    elif l==1 and len(term) <8:
        threshold=81
    return(threshold)
    
def clean_term (term):
    return (re.sub(r'[\.\,\!\?\"\'\n$]+','',term))

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
         
def search_dictionary(targets, d):
   likely_matches = []
   for target in targets:
     threshold = get_threshold(target, 1)
     for entry, definition in d.items():   
       if (fuzz.ratio(target, entry) > threshold ): 
          match = Match(target, entry, definition, threshold)
          likely_matches.append(match)
   return (likely_matches)

def get_sentences(content): 
   return (nltk.sent_tokenize(content))

def get_words(content):
   return (nltk.word_tokenize(content))


def get_sentence_index(s, all_text):
    return (all_text.index(s))

def get_index(term, s):
   sen= get_sentence_index(s, text)
   return (s.find(term)+sen)

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

def get_dictionary_ngrams(rows, n):
    terms ={}
    if n == 2:
       pattern = bigram_pattern
    elif n == 3:
       pattern = trigram_pattern
    elif n ==4:
       pattern = quadgram_pattern
    for key,value in rows.items():
       match= re.match(pattern,key)
       if match:
           terms[key]=value
    return (terms)    

def get_three_predictions(model, sentence):
      if (sentence):
          cand1 = model.wv.doesnt_match(sentence)
          sentence_v2 = remove_sentence_word(cand1,sentence)
          if (sentence_v2):
              cand2 = model.wv.doesnt_match(sentence_v2)
              sentence_v3 = remove_sentence_word(cand2,sentence)
              if (sentence_v3):
                  cand3 = model.wv.doesnt_match(sentence_v3)
                  return (cand1, cand2, cand3)
              else:
                  return(cand1, cand2) 
          else:
              return (cand1)
      else:
          return (None)

  
def merge_dictionary(x,y):
    z=x.copy()
    z.update(y)
    return z

def check_samelabels (new_label, labels):
    found = False
    for current_label in labels: 
       if new_label == current_label: 
           print ("already have a label with " + new_label.label + " at position ",current_label.pos)
           return True
    return False

def read_dictionary():
    dict_terms ={}
    with open(dict_path, "r") as f:
         reader = csv.reader(f, delimiter='\t')
         for key,value in reader:
            dict_terms[key]=value;
    f.close()
     
    my_dict = merge_dictionary(get_dictionary_ngrams(dict_terms,4),get_dictionary_ngrams(dict_terms,3))
    my_dict = merge_dictionary(my_dict,get_dictionary_ngrams(dict_terms,2))
    return (my_dict)

def fuzzy_label(str_a, str_b, orig_str):
    test = True
    i=0
    labels = []
    l = len(str_a.split()) 
    splitted = re.findall(punc_pattern,orig_str) 
    extent = (len(splitted)-l+1) 
    while (i < extent and test):
        test = clean_term(" ".join(splitted[i:i+l])).lower()  
        for thresh in ngram_thresholds:
          if fuzz.ratio(str_a, test)> thresh:
            label = phraseLabel(str_a, orig_str.index(test), len(test), str_b, test, thresh)
            if check_samelabels(label, labels) == False:
                 labels.append(label)
        i=i+1
    if (labels):
        return(labels)
    else:
        return(None)

def collect_ngrams(text):
    for s,t in dict_grams.items():
        labels = fuzzy_label(s.lower(), t, text)
        if (labels):
            all_labels.extend(labels)
    if (all_labels):
    	json_string = json.dumps([ob.__dict__ for ob in all_labels])
   	return (json_string)
    else:
        print ("there were no known ngrams in the source text")


def collect_unigrams(text):
  for s in get_sentences(text):
    splits=(s.split(' '))
    m1_candidates=[]
    m2_candidates=[]
    sent = remove_stopwords(s)
    try:
        #pred1 = get_three_predictions(model1, sent)
        #pred2 = get_three_predictions(model2, sent)
        #common =list(set(pred1).intersection(pred2))
        common = [splits[4]]
    except Exception as e:
        print ("the error in running models: ",e)
        continue;
    if (common):
        matches = search_dictionary(common, get_dictionary_unigrams())
    else:
        continue;
    if bool(matches): 
      for m in matches:
          index=get_index(m.candidate,s)
          label=phraseLabel (m.dict_term, index, len(m.candidate), m.definition, m.candidate, m.thresh )
          all_labels.append(label)
    else:
        continue;
  json_string = json.dumps([ob.__dict__ for ob in all_labels])
  return(json_string)


text= read_file()
dict_grams = read_dictionary()
all_labels = []
first =  (collect_ngrams(text))
second = json.loads(collect_unigrams(text))

print json.dumps(second, indent=2, sort_keys=True) 



