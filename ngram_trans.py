import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import csv
import re
import string
from fuzzywuzzy import fuzz
from nltk.stem.porter import PorterStemmer
from colorama import init, Fore, Back, Style
from ppretty import ppretty
init(convert=True)
import warnings
warnings.filterwarnings("ignore")

file_path = "input_text.txt"
dict_path = "dictionary.tsv"
punc_pattern = r'[\S\n\t\r]+'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'


class phraseLabel :
    def __init__(self, label, pos, length, defin, found, thresh):
        self.label = label
        self.pos = pos
        self.thresh = thresh
 	self.length = length
 	self.defin = defin
        self.found = found

    def __eq__(self, other):
        return (self.label in other.label  and self.thresh == other.thresh)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__ (self):
         return hash((self.label,self.pos))


def read_file():
  with open(file_path, 'r') as ff:
    content = ff.read()
  ff.close()
  return(content)

def clean_term (term):
    return (re.sub(r'[\.\,\!\?\"\'\n$]+','',term)) 
  
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


def fuzzy_label(str_a, str_b, orig_str):
    test = True
    i=0
    labels = []
    l = len(str_a.split()) 
    thresholds = [87]
    splitted = re.findall(punc_pattern,orig_str) 
    extent = (len(splitted)-l+1) 
    while (i < extent and test):
        test = clean_term(" ".join(splitted[i:i+l])).lower()  
        for thresh in thresholds:
          if fuzz.ratio(str_a, test)> thresh:
            label = phraseLabel(str_a, i, l, t, test, thresh)
            if check_samelabels(label, labels) == False:
           	 labels.append(label)
        i=i+1
    if (labels):
	return(labels)
    else:
	return(None)
    

text= read_file()
dict_grams = read_dictionary()
all_labels = []

for s,t in dict_grams.items():
   labels = fuzzy_label(s.lower(), t, text)
   if (labels):
      all_labels.extend(labels)

for l in all_labels:
   print (ppretty(l))


