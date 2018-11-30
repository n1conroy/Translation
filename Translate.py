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
#import enchant

import warnings
warnings.filterwarnings("ignore")


thresholds =[85]

dict_path = "C:\develop\DataScienceMaster\Translate\data\dictionary.tsv"
#dict_path = "C:\develop\DataScienceMaster\Translate\data\gangslang.tsv"
#englishDictionary = enchant.Dict("en_US")

punc_pattern = r'[\S\n\t\r]+'
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
stops= set(stopwords.words('english'))

file_path = "C:\develop\DataScienceMaster\Translate\data\input_text.txt"


model1 = Word2Vec.load("saved_models/embeddings3.model")
#model2 = gensim.models.KeyedVectors.load_word2vec_format('saved_models\embeddingmodel6.bin', binary=True, limit=500000)
model2 = Word2Vec.load("saved_models/embeddings5.model")

class phraseLabel :
    def __init__(self, label, pos, length, defin, found, thresh):
        self.label = label
        self.pos = pos
        self.thresh = thresh
        self.length = length
        self.defin = defin
        self.found = found

class Match(object):
    def __init__(self, candidate, dict_term, definition):
        self.candidate=candidate
        self.dict_term=dict_term
        self.definition=definition

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
         
def search_dictionary(targets, d, threshold):
   likely_matches = []
   for target in targets:
     threshold = get_threshold(target, 1)
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
    for current_label in labels:
        if new_label == current_label: 
           print ("already have a label with " + current_label.label + " at position ",current_label.pos)
           return True
    return False

def read_dictionary():
    dict_terms ={}
    with open(dict_path, "r", encoding="utf-8") as f:
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
        for thresh in thresholds:
          if fuzz.ratio(str_a, test)> thresh:
            label = phraseLabel(str_a, i, l, str_b, test, thresh)
            if check_samelabels(label, labels) == False:
           	 labels.append(label)
        i=i+1
    if (labels):
        return(labels)
    else:
        return(None)

def collect_ngrams():
    for s,t in dict_grams.items():
        labels = fuzzy_label(s.lower(), t, text)
        if (labels):
            all_labels.extend(labels)
    if (all_labels):
        for l in all_labels:
            print (ppretty(l))
    else:
        print ("there were no known ngrams in the source text")

def collect_unigrams():
  for s in get_sentences(text):
   
    m1_candidates=[]
    m2_candidates=[]
    sent = remove_stopwords(s)
    print ("the sent is ",sent)
    try:
        pred1 = get_three_predictions(model1, sent)
        pred2 = get_three_predictions(model2, sent)
        common =list(set(pred1).intersection(pred2))
    except Exception as e:
        print ("the error in running models: ",e)
        continue;
    if (common):
        print ("for sentence:", s, "the words in common are ", common)
        matches = search_dictionary(common, get_dictionary_unigrams(), thresholds)
    else:
        continue;
    if bool(matches): 
      for m in matches:
          print ("The dicionary defines ", m.dict_term, "as", m.definition)
          index=get_index(m.candidate,s)
          #add_label(s, candidate, index, confidence, length=1)
    else:
       print ("but, there was nothing resembling those terms found in the slang dictionary\n")

	    
text= read_file()
dict_grams = read_dictionary()
all_labels = []
#collect_ngrams()
collect_unigrams()
