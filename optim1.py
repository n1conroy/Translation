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
from HoverInfo import HoverInfo
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

def read_file(file_path):
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
    return (dict_terms)



def substring_indexes(substring, string):
    last_found = -1 
    while True:
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:  
            break  
        yield last_found

def collect_ngrams(orig_str):
    grams = [2, 3, 4]
    orig = orig_str.lower()
    splitted = re.findall(punc_pattern, orig_str) 
    for g in grams:
      i=0
      extent = (len(splitted)-g+1)
      while (i < extent):
        test = clean_term(" ".join(splitted[i:i+g])).lower()  
        print (test)
        for thresh in ngram_thresholds:
	    if test in dict_grams:
              defin = dict_grams[test]	     
              indexes = list(substring_indexes(test, orig_str))
	      for ind in indexes:
       	        label = phraseLabel(test, ind, len(test), defin, test, thresh)
	        if check_samelabels(label, all_labels) == False:
                  all_labels.append(label)
        i=i+1
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
        if (sent): 
		pred1 = get_two_predictions(model1, sent)
	        print ('pred1', pred1, s)
        if (sent_prev): 
		pred2 = get_two_predictions(model2, sent_prev)
		print ('pred2', pred2, s)
        common = list(set(pred1).intersection(pred2))
    except Exception as e:
        print ("Error in running models: ",e, "with ", sent, sent_prev)
        continue;
    if (common):
       for c in common:
	  for thresh in ngram_thresholds:	
	    if c in dict_grams:
               defin = dict_grams[c]
	       i = get_index(c, s)
	       label=phraseLabel (dict_grams[c], i, len(c), defin, c, thresh )
	       if check_samelabels(label, all_labels) == False:
                   all_labels.append(label)
    else:
        continue
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
         if w in dict_grams:
           print (w)
           for thresh in ngram_thresholds:
              defin = dict_grams[w]
              indexes = list(substring_indexes(w, texts))
	      for ind in indexes:
       	        label = phraseLabel(w, ind, len(w), defin, w, thresh)
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
    import sys

    filename = str(sys.argv[1])
    texts= read_file(filename)
    dict_grams=read_dictionary()
    all_labels = []
    all_found_ngrams = (collect_ngrams(texts))
    all_found_unigrams = (collect_unigrams(texts))
    all_found_noneng = (collect_non_dict(texts))
    all_labels.sort(key=operator.attrgetter("pos"),reverse=False)

    if the slang is less than 3% it's not useful
      
    if (len (all_labels)/len(texts.split(' ')) < 0.03): 
       #all_labels=[]
    if all_labels:
       all_lab = sorted(all_labels, key=lambda l: l.pos, reverse=True)
       root = Tk()
       root.geometry('1200x700')
       Visual(root).pack(side="top", fill="both", expand="true")
       root.mainloop()
    else:
       print ("We found no slang to report")
    if (all_found_unigrams!='There were no unigrams in the source text.'):
      print ("The ngrams and unigrams json is ", json.loads(all_found_noneng))
    else:
      #print (all_found_ngrams+'\n')
      print (all_found_unigrams+'\n')
    

   

