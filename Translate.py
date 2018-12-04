#!/usr/bin/python
import csv
import re
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import fuzz
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from ppretty import ppretty
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


#model1 = Word2Vec.load("saved_models/embeddings4.model")
model2 = gensim.models.KeyedVectors.load_word2vec_format('saved_models/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
model1 = gensim.models.KeyedVectors.load_word2vec_format('saved_models/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
#model2 = Word2Vec.load("saved_models/embeddings5.model")

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
    	return(json_string)
    else:
        return ("There were no known ngrams in the source text.")


def collect_unigrams(texts):
  to_check = []
  for s in get_sentences(texts):
    to_check=set(to_check)
    m1_candidates=[]
    m2_candidates=[]
    sent = remove_stopwords(s)
    try:
        #pred1 = get_two_predictions(model1, sent)
        #pred2 = get_two_predictions(model2, sent)
        #common = list(set(pred1).intersection(pred2))
        common =s[4]
        if (to_check):
            common.extend(to_check)
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
                     label = phraseLabel(key, ind, len(w), val, w, 85)                         
                     if check_samelabels(label, all_labels) == False:
	             	all_labels.append(label)
    if (all_labels):
	json_string = json.dumps([ob.__dict__ for ob in all_labels])
        return(json_string)
    else:
      return ("No non English words were found in the dictionary ")


texts= read_file()
dict_grams = read_dictionary()
all_labels = []


all_found_ngrams = (collect_ngrams(texts))
all_found_unigrams = (collect_unigrams(texts))
all_found_noneng = (collect_non_dict(texts))




all_labels.sort(key=operator.attrgetter("pos"),reverse=False)


def click(self, event):
    # get the index of the mouse click
    index = self.MT.index("@%s,%s" % (event.x, event.y))

        # get the indices of all "adj" tags
    tag_indices = list(self.MT.tag_ranges('adj'))

        # iterate them pairwise (start and end index)
    for start, end in zip(tag_indices[0::2], tag_indices[1::2]):
        # check if the tag matches the mouse click index
        if self.MT.compare(start, '<=', index) and self.MT.compare(index, '<', end):
            # return string between tag start and end
            return (start, end, self.MT.get(start, end))


if all_labels:
   all_lab = sorted(all_labels, key=lambda l: l.pos, reverse=True)
   if (all_found_ngrams !='There were no known ngrams in the source text.'):
      print ("The ngrams json is ", json.loads(all_found_ngrams))
   if (all_found_unigrams!='There were no unigrams in the source text.'):
      print ("The unigrams json is ", json.loads(all_found_unigrams))
   window = Tk()
   window.geometry('1100x600')
   text = Text(window, width=100, height=300)
   text.insert('1.0', texts)
   text.pack(side="left") 
   for l in all_lab:
      newcolor = colors[randint(0,12)]
      start ="1.0+"+str(l.pos)+"chars"
      duration = l.pos+l.length
      end = "1.0+"+str(duration)+"chars"
      text.tag_configure("highlight", background=newcolor)
      text.tag_add("highlight",start, end)
      bu1=Button(window, text='TERM: <'+str(l.input_term)+'> at POSITION: <'+str(l.pos)+'>', width=10, bg=newcolor, fg='black')
      lstring = 'DICTIONARY: '+str(l.dict_term)+'\nCONFIDENCE: '+str(l.thresh)+'\nDEFINITION: '+str(l.defin) 
      bu1.submitButton =  HoverInfo(bu1, lstring)  
      bu1.pack(fill=X, pady=7, padx=2)
   window.mainloop()
else:
   print (all_found_ngrams+'\n')
   print (all_found_unigrams+'\n')






