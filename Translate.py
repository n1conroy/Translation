#!/usr/bin/python
import csv
import re
import json
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import fuzz
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from Tkinter import *  
from nltk.corpus import words
from random import randint
import operator

import warnings
warnings.filterwarnings("ignore")

ps = nltk.stem.SnowballStemmer('english')
ngram_thresholds =[85]

#dict_path = "C:\develop\DataScienceMaster\Translate\data\dictionary.tsv"
dict_path = "dictionary.tsv"


symb_pattern = r'[\<\>\{\}\.\,\!\?\"\n$]+'
punc_pattern = r'[\S\t\r]+'
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
stops= set(stopwords.words('english'))

#file_path = "data/input_text.txt"
file_path = "input_text.txt"

colors=['sea green', 'maroon3', 'light salmon', 'slate blue', 'turquoise1','RoyalBlue1', 'coral', 'khaki1','ivory3','slate grey', 'yellow2', 'red3', 'purple']


model1 = Word2Vec.load("saved_models/embeddings4.model")
model2 = Word2Vec.load("saved_models/embeddings5.model")
#model2 = gensim.models.KeyedVectors.load_word2vec_format('saved_models/GoogleNews-vectors-negative300.bin', binary=True, limit=90000)

class phraseLabel :
    def __init__(self, dict_term, pos, length, defin, input_term, thresh):
        self.dict_term = dict_term
        self.pos = pos
        self.thresh = thresh
        self.length = length
        self.defin = defin
        self.input_term = input_term
    
    def __eq__(self, other):
        return (self.dict_term == other.dict_term and self.pos == other.pos and self.thresh == other.thresh)

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



class Visual(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        self.t1 = Text(self, width=100, height=300)
        self.t1.insert('1.0', texts)
        self.t1.pack(side="left")
        for l in all_lab:
          newcolor = colors[randint(0,12)]
          start ="1.0+"+str(l.pos)+"chars"
          duration = l.pos+l.length
          end = "1.0+"+str(duration)+"chars"
          self.t1.tag_configure(l.dict_term, background=newcolor)
          self.t1.tag_add(l.dict_term, start, end)
          self.t1.tag_bind(l.dict_term,"<Button-1>", lambda event, dict_term=str(l.dict_term), thresh=str(l.thresh), defin=str(l.defin): self.on_click(dict_term, thresh, defin))     
          self.t1.tag_bind(l.dict_term,"<Leave>", self.on_leave)
          self.t1.tag_bind(l.dict_term,"<Enter>", self.on_enter)
         
        self.l2 = Label(self, text="DETAILS", width=40)
        self.l2.pack(side="top", fill="x")

    def on_click(self, dict_term, thresh, defin):
        self.l2.configure(text='DICTIONARY: '+dict_term+'\nCONFIDENCE: '+thresh+'\nDEFINITION: '+defin) 

    def on_leave(self, event):
        self.l2.configure(text="")

    def on_enter(self, event):
        self.t1.config(cursor="arrow")


def get_threshold (term, l):
    threshold=80
    if l ==1 and len(term) <5:
        threshold =87
    elif l == 1 and len(term) <6 :
        threshold=86
    elif l ==1 and len(term) <7:
        threshold=82
    elif l==1 and len(term) <8:
        threshold=79
    return(threshold)
    
def clean_term (term):
    return (re.sub(symb_pattern,'',term))

def read_file():
  with open(file_path, 'r') as ff:
  #with open(file_path, 'r', encoding='utf-8') as ff:
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

def is_English(word):
    if not (clean_term(word.lower()) in words.words() or clean_term(word.lower()) in stops):
       return(False)
    else:
       return(True)
        
def search_dictionary(targets, d):
   likely_matches = []
   for target in targets:
     target=clean_term(target)
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
   sen= get_sentence_index(s, texts)
   return (s.find(term)+sen)

def get_dictionary_unigrams():
    dict_terms ={}
    with open(dict_path, "r") as f:
    #with open(dict_path, "r", encoding='utf-8') as f:
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

def get_two_predictions(model, sentence):
    if (sentence):
      cand1 = model.wv.doesnt_match(sentence)
      sentence_v2 = remove_sentence_word(cand1,sentence)
      if (sentence_v2):
          cand2 = model.wv.doesnt_match(sentence_v2)
          return (cand1, cand2)
      else:
          return(cand1) 
    else:
        return(None)
    
  
def merge_dictionary(x,y):
    z=x.copy()
    z.update(y)
    return z

def check_samelabels (new_label, labels):
    found = False
    for current_label in labels: 
       if new_label == current_label: 
           return True
    return False

def read_dictionary():
    dict_terms ={}
    with open(dict_path, "r") as f:
    #with open(dict_path, "r", encoding='utf-8') as f:
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
             orig = orig_str.lower()
             label = phraseLabel(str_a, texts.lower().index(test), len(test), str_b, test, thresh)
             if check_samelabels(label, labels) == False:
                  labels.append(label)
        i=i+1
    if (labels):
        return(labels)
    else:
        return(None)

def collect_ngrams(texts):
    for s,t in dict_grams.items():
        labels = fuzzy_label(s.lower(), t, texts)
        if (labels):
            all_labels.extend(labels)
    if (all_labels):
      	json_string = json.dumps([ob.__dict__ for ob in all_labels])
      	return (json_string)
    else:
        return ("There were no known ngrams in the source text.")


def collect_unigrams(texts):
  for s in get_sentences(texts):
    sent = remove_stopwords(s)
    try:
	sent_prev = sent
        if (sent): pred1 = get_two_predictions(model1, sent)
	else:
	   print ("empty sentence - moving on!")
        if (sent_prev):pred2 = get_two_predictions(model2, sent_prev)
        else:
	   print ("empty sentence - moving on!")
	print (sent, sent_prev)
        common = list(set(pred1).intersection(pred2))
    except Exception as e:
        print ("Error in running models: ",e)
        continue;

    if (common):
        matches = search_dictionary(common, get_dictionary_unigrams())
    else:
        continue;
    if bool(matches): 
      for m in matches:
          index=get_index(m.candidate,s)
          label=phraseLabel (m.dict_term, index, len(m.candidate), m.definition, m.candidate, m.thresh )
          if check_samelabels(label, all_labels) == False:
              all_labels.append(label)
    else:
         continue;
  if (all_labels):
      json_string = json.dumps([ob.__dict__ for ob in all_labels])
      return(json_string)
  else:
      return ("There were no unigrams in the source text.")


def collect_non_dict(texts):
    terms= get_dictionary_unigrams()
    i=0
    splitted = re.findall(punc_pattern,texts) 
    for w in splitted: 
        i = i+1   
        w=clean_term(w)   
        if is_English(w) == False:
           for key, val in terms.items():
            threshold = get_threshold(key, 1)
            if fuzz.ratio(key, w) > threshold:
                index = [m.start() for m in re.finditer(w, texts)]
                for ind in index:                  
                    label = phraseLabel(key, ind, len(w), val, w, threshold)                         
                    if check_samelabels(label, all_labels) == False:
                        all_labels.append(label)
    if (all_labels):
     	json_string = json.dumps([ob.__dict__ for ob in all_labels])
     	return(json_string)
    else:
      return ("No non English words were found in the dictionary ")


def get_sentiment(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score



if __name__ == "__main__":
    texts= read_file()
    dict_grams = read_dictionary()
    all_labels = []
    all_found_ngrams = (collect_ngrams(texts))
    all_found_unigrams = (collect_unigrams(texts))
    all_found_noneng = (collect_non_dict(texts))
    all_labels.sort(key=operator.attrgetter("pos"),reverse=False)

    if all_labels:
      all_lab = sorted(all_labels, key=lambda l: l.pos, reverse=True)
    if (all_found_unigrams!='There were no unigrams in the source text.'):
      print ("The ngrams and unigrams json is ", json.loads(all_found_noneng))
    else:
      print (all_found_ngrams+'\n')
      print (all_found_unigrams+'\n')
    

    root = Tk()
    root.geometry('1200x700')
    Visual(root).pack(side="top", fill="both", expand="true")
    root.mainloop()







