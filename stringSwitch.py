import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import csv
import re
import string
from fuzzywuzzy import fuzz
from nltk.stem.porter import PorterStemmer
from colorama import init, Fore, Back, Style
init(convert=True)
file_path = "C:\develop\DataScienceMaster\Translate\input_text.txt"
dict_path = "C:\develop\DataScienceMaster\Translate\dictionary.tsv"
punc_pattern = r'[\S\n\t\r]+'
unigram_pattern= r'^[a-zA-Z0-9\-\']+$'
bigram_pattern=r'^[a-zA-Z0-9\'\-]+\s[a-zA-Z0-9\'\-]+$'
trigram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
quadgram_pattern=r'^[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+\s[a-zA-Z0-9\']+$'
porter_stemmer = PorterStemmer()              



def read_file():
  with open(file_path, 'r') as ff:
    content = ff.read()
  ff.close()
  return(content)

  
def read_dictionary():
    dict_terms ={}
    with open(dict_path, "r") as f:
         reader = csv.reader(f, delimiter='\t')
         for key,value in reader:
            dict_terms[key]=value;
    f.close()
     
    my_dict = merge_dictionary(get_dictionary_ngrams(dict_terms,4),get_dictionary_ngrams(dict_terms,3))
    my_dict = merge_dictionary(my_dict,get_dictionary_ngrams(dict_terms,2))
    #my_dict = merge_dictionary(my_dict,get_dictionary_ngrams(dict_terms,1))
    return (my_dict)

def get_dictionary_ngrams(rows, n):
    terms ={}
    if n == 1:
       pattern = unigram_pattern
    elif n == 2:
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
  
def get_pos_tags(text):
    return (nltk.pos_tag(word_tokenize(text)))

def get_term_pos(tagged, term):
    for key,value in tagged:
         if key == term:
             return (value)
        
def clean_term (term):
    return (re.sub(r'[\.\,\!\?\"\'\n$]+','',term))    


def get_threshold (term, l):
    threshold=80
    if l ==1 and len(term)<5 :
        threshold =89
    elif l == 1 and len(term) <6 :
        threshold=88
    elif l ==1 and len(term)>5:
        threshold=83
    elif l==2 and len(term)<8:
        threshold=81
    elif l==2 and len(term)>7:
        threshold=80
    elif l==3:
        threshold=75
    return(threshold)

def fuzzy_replace(str_a, str_b, orig_str):
    flag = False
    test = True
    i=0
    l = len(str_a.split()) # adjust threshold based on size and number of terms
    threshold = get_threshold(str_a, l)
    splitted = re.findall(punc_pattern,orig_str) # break up the original text, keeping all the spacing
    extent = (len(splitted)-l+1) # get the length of the source text
    while (i < extent and test):
        test = clean_term(" ".join(splitted[i:i+l])).lower() #test is the currently moving term window 
        before = " ".join(splitted[:i])  #get all the text before the match from the original text
        after = " ".join(splitted[i+l:])  
        i=i+1
        if (not((after.find('>>') < after.find('<<')) and (before.find('>>') < before.find('<<')))) : # we are not currently in a tag, so we can try matching
          if (fuzz.ratio(str_a, test) > threshold ) and ('<<' not in test and '>>' not in test):  #if the target closely matches the current ngram
             print ('the test is ',test)
             print ('the match is ',str_a)
             if (flag==True):
                 splitted=changed_string.split()
             changed_string = before+" "+str_b+" "+after   #output the sandwich of these three strings
             i=i+l
             flag=True
        else:
          continue
    if (flag == False):
        return (" ".join(splitted))
    else:
        return (changed_string)
 
text= read_file()
print ("\nTHE ORIGINAL VERSION IS:\n",text)
pos_tags=get_pos_tags(text)
dict_grams = read_dictionary()
current_old = text

for s,t in dict_grams.items():
   target = Fore.YELLOW+'<<TR:'+s+':'+Fore.RED+t+'>>'+Fore.WHITE
   current = fuzzy_replace(s, target ,current_old)
   current_old = current
     
print ("\n\nTHE TRANSLATED VERSION IS:\n",current_old)


