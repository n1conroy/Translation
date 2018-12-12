#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import csv
import re
import json
import nltk
import gensim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import process, fuzz
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from tkinter import *  
from nltk.corpus import words
from random import randint
import operator
import time

nltk_sentiment = SentimentIntensityAnalyzer()
ngram_thresholds =[85]
uni_terms ={}
bi_terms = {}
tri_terms = {}
quad_terms = {}
dict_path = "C:\develop\DataScienceMaster\Translate\data\dictionary.tsv"

symb_pattern = r'[\<\>\{\}\.\,\!\?\"\n$]+'
punc_pattern = r'[\S\t\r]+'
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
stops= set(stopwords.words('english'))
colors=['sea green', 'maroon3', 'light salmon', 'slate blue', 'turquoise1','RoyalBlue1', 'coral', 'khaki1','ivory3','slate grey', 'yellow2', 'red3', 'purple']
colors={'neg':'red3','pos':'green','neu':'light blue', 'compound':'orange', 'None':'yellow'}
model1 = Word2Vec.load("saved_models/embeddings4.model")
model2 = Word2Vec.load("saved_models/embeddings5.model")
engs = words.words()+list(stops)

unigram_thresholds = {1:100, 2:100, 3:97, 4:95, 5:91, 6:86, 7:82, 8:80, 9:79, 10:77, 11:75, 12:74, 13:73}

#model1 = gensim.models.KeyedVectors.load_word2vec_format('saved_models/embeddings6-vectors-negative300.bin', binary=True, limit=50000)

class phraseLabel :
    def __init__(self, dict_term, position, length, defin, input_term, emot, thresh):
        self.dict_term = dict_term
        self.position = position
        self.length = length
        self.defin = defin
        self.input_term = input_term
        self.emot = emot
        self.thresh = thresh
    
    def __eq__(self, other):
        return (self.dict_term == other.dict_term and self.position == other.position and self.position != -1)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__ (self):
         return hash((self.label,self.position))


class Visual(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        
        self.t1 = Text(self, width=100, height=300)
        self.t1.insert('1.0', texts)
        self.t1.pack(side="left")
        for l in all_lab:
          start ="1.0+"+str(l.position)+"chars"
          duration = l.position+l.length
          end = "1.0+"+str(duration)+"chars"
          self.t1.tag_configure(l.dict_term, background=colors[str(l.emot)])
          self.t1.tag_add(l.dict_term, start, end)
          print (l.input_term, l.position)
          self.t1.tag_bind(l.dict_term,"<Button-1>", lambda event, dict_term=str(l.dict_term), thresh=str(l.thresh), defin=str(l.defin), position=str(l.position), emot= str(l.emot): self.on_click(dict_term, thresh, defin, position, emot))     
          self.t1.tag_bind(l.dict_term,"<Leave>", self.on_leave)
          self.t1.tag_bind(l.dict_term,"<Enter>", self.on_enter)
         
        self.l3 = Label(self, text="Details", width=40)
        self.l2 = Label(self, text="", width=40)
        self.l3.pack(side="top", fill="x") 
        self.l2.pack(side="top", fill="x")

    def on_click(self, dict_term, thresh, defin, position, emot):
        self.l2.configure(text='DICTIONARY: '+dict_term+'\nCONFIDENCE: '+thresh+'\nDEFINITION: '+defin+'\nLOCATION: '+position+'\nSENTIMENT:'+emot) 

    def on_leave(self, event):
        self.l2.configure(text="")

    def on_enter(self, event):
        self.t1.config(cursor="arrow")


def get_threshold(term):
    try:
        thresh = unigram_thresholds[len(term)]
        if (thresh):
           return (thresh)
    except:
        return (90)
  

def clean_term (term):
    return (re.sub(symb_pattern,'',term))

def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8', errors="ignore") as ff:
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
    t = clean_term(word.lower())
    if t in engs:
       return(True)
    else:
       return(False)
        

def get_sentences(content): 
   natural_count = nltk.sent_tokenize(content)
   if (len(natural_count) < 3):
     #return content.split('\n')
     return (natural_count)
   else:
     return (natural_count)

def get_words(content):
   return (nltk.word_tokenize(content))

def get_sentence_index(s, all_text):
    return (all_text.index(s))

def get_index(term, s):
   sen= get_sentence_index(s, texts)
   return (s.find(term)+sen)

def get_dictionary_unigrams():
    dict_terms ={}
    #with open(dict_path, "r") as f:
    with open(dict_path, "r", encoding='utf-8') as f:
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

def get_dictionary_grams():
    with open(dict_path, "r", encoding='utf-8') as f:
         reader = csv.reader(f, delimiter='\t')
         for key,value in reader:
            if (re.match(unigram_pattern,key)):
                uni_terms[key]=value;
            elif (re.match(bigram_pattern, key)):
                bi_terms[key]=value;
            elif (re.match(trigram_pattern, key)):
                tri_terms[key]=value;
            elif (re.match(quadgram_pattern, key)):
                quad_terms[key]=value;
    f.close()


def check_phrase_candidate(phrase, texts):
   scored = phrase.replace(" ","_")
   sents = get_sentences (texts)
   for s in sents:
       words = get_words(s)
       if phrase in s:
          for t in phrase:
              s = remove_sentence_word(t, words)
          try:
              rated = model3.self.doesnt_match(words.append(scored))
              if scored == rated:  
                 return (True)
          except Exception as e:
              print ("Issue with the phrase check", e)
   return(False)



def substring_indexes(substring, string):
    last_found = -1 
    while True:
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:  
            break  
        yield last_found

def collect_ngrams(orig_str):
    grams = {2:bi_terms, 3:tri_terms, 4:quad_terms}
    orig = orig_str.lower()
    splitted = re.findall(punc_pattern, orig_str) 
    for g in grams.keys():
      i=0
      extent = (len(splitted)-g+1)
      choices = list(grams[g])
      while (i < extent):
        test = clean_term(" ".join(splitted[i:i+g])).lower()  
        for thresh in ngram_thresholds:
            x= (process.extractOne (test, choices, scorer=fuzz.ratio, score_cutoff = thresh))
            if (x):
              found = x[0]
              defin = grams[g][found]	     
              indexes = list(substring_indexes(test, orig_str))
              for ind in indexes:
                emot = get_sentiment(orig_str[ind-20:ind+20])
       	        label = phraseLabel(x[0], ind, len(test), defin, test, emot, thresh)
       	        if check_samelabels(label, all_labels) == False:
                  all_labels.append(label)
        i=i+1
    if (all_labels):
      	json_string = json.dumps([ob.__dict__ for ob in all_labels])
      	return (json_string)
    else:
        return ("There were no known ngrams in the source text.")

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def collect_unigrams(texts):
  choices = list(uni_terms)
  for s in get_sentences(texts):
    sent = remove_stopwords(s)
    sent_prev = sent
    if len(sent) < 3:
        continue;
    elif (len(sent) < 6 and len(sent) > 1):
       common = [model2.wv.doesnt_match(sent)]
    else:
       try:
           pred1 = get_two_predictions(model1, sent)
           pred2 = get_two_predictions(model2, sent_prev)
           common = list(set(pred1).intersection(pred2))
       except Exception as e:
           print ("Error in running models: ",e, "with ", sent)
           continue;
    if (common):
       for c in common:
          if c in choices:
              defin = uni_terms[c]
              thresh = 100
              i = get_index(c, s)
              emot = get_sentiment (s)
              label=phraseLabel (c, i, len(c), defin, c, emot, thresh)
              if check_samelabels(label, all_labels) == False:
                     all_labels.append(label)
          else:
              thresh = get_threshold(c)
              x = process.extractOne(c, choices, scorer=fuzz.ratio, score_cutoff = thresh )
              if (x):
                  defin = uni_terms[x[0]]
                  i = get_index(c, s)
                  emot = get_sentiment (s)
                  label=phraseLabel (x[0], i, len(c), defin, c, emot, thresh)
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
    choices = list(uni_terms)
    i=0
    splitted = re.findall(punc_pattern,texts) 
    for w in splitted: 
       i = i+1   
       w=clean_term(w)   
       if is_English(w) == False:
          thresh = get_threshold(w) 
          x = process.extractOne(w, choices, scorer=fuzz.ratio, score_cutoff = thresh )
          if (x): # found the non English word in our dictionary
             indexs = list(find_all(texts, w))
             defin = uni_terms[x[0]]
             for ind in indexs:                  
                emot = get_parent_sentiment(w, splitted[i-6:i+6])
                label = phraseLabel(x[0], ind, len(w), defin, w, emot, thresh)                         
                if check_samelabels(label, all_labels) == False:
                     all_labels.append(label)
    if (all_labels):
     	json_string = json.dumps([ob.__dict__ for ob in all_labels])
     	return(json_string)
    else:
      return ("No non English words were found in the dictionary ")


def collect_non_dict2(texts):
    choices = list(uni_terms)
    splitted = set(re.findall(punc_pattern, texts))
    found = list (splitted-(set(engs)))
    for f in found:
        thresh = get_threshold(f)
        x = process.extractOne(f, choices, scorer=fuzz.ratio, score_cutoff =thresh)
        if (x):
            indexes = list(find_all(texts, f))
            defin = uni_terms[x[0]]
            for ind in indexes:                  
                emot = get_sentiment(texts[ind-20:ind+20])
                label = phraseLabel(x[0], ind, len(f), defin, f, emot, thresh)                         
                if check_samelabels(label, all_labels) == False:
                     all_labels.append(label)
    if (all_labels):
     	json_string = json.dumps([ob.__dict__ for ob in all_labels])
     	return(json_string)
    else:
      return ("No non English words were found in the dictionary ")


def get_sentiment(sentence):
    score = nltk_sentiment.polarity_scores(sentence)
    max_emot = max(score.items(), key=operator.itemgetter(1))[0]
    if (max_emot):
       return (max_emot)
    else:
       return(None)


    return score



if __name__ == "__main__":
    import sys    

    try:
        filename = str(sys.argv[1])
    except:
        print ("Please provide the path of a text file for input")
        exit()

    all_labels = []
    texts= read_file(filename)
    startTime = time.time()
    get_dictionary_grams()
    dict_time = time.time() - startTime
    print ('DICTIONARY_TIME', dict_time)
    
    startTime = time.time()
    all_found_ngrams = (collect_ngrams(texts))
    ngram_time = time.time() - startTime
    print ('NGRAM_TIME', ngram_time)
    
    startTime = time.time()
    all_found_unigrams = (collect_unigrams(texts))
    unigram_time = time.time() - startTime
    print ('UNIGRAM_TIME', unigram_time)
    startTime = time.time()
    all_found_noneng = (collect_non_dict2(texts))
    noneng_time = time.time() - startTime
    print ('NONENG_TIME', noneng_time)
    
    all_labels.sort(key=operator.attrgetter("position"),reverse=False)

    #if the slang is less than 3% it's not useful
    if (len (all_labels)/len(texts.split(' ')) < 0.03): 
       all_labels=[]
       print ("We found no significant slang to report")
    else:
       all_lab = sorted(all_labels, key=lambda l: l.position, reverse=True)
       root = Tk()
       root.title ("Slang Finder")
       root.geometry('1200x700')
       Visual(root).pack(side="top", fill="both", expand="true")
       root.mainloop()
       if (all_found_unigrams!='There were no unigrams in the source text.'):
          print ("The ngrams and unigrams json is ", json.loads(all_found_noneng))
       else:
          print (all_found_ngrams+'\n')
          print (all_found_unigrams+'\n')
