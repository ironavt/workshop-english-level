# this .py file contains functions I use in this project
'''
- - - - -
Imports
- - - - -
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import os
import pysrt # https://github.com/byroot/pysrt
import chardet
import codecs

# import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
- - - - -
Globals
- - - - -
'''

HTML = re.compile(r'<.*?>')
TAG = re.compile(r'{.*?}')
COMMENTS = re.compile(r'[\(\[][A-Z ]+[\)\]]')
LETTERS = re.compile(r'[^a-zA-Z\'.,!? ]')
SPACES = re.compile(r'([ ])\1+')
DOTS = re.compile(r'[\.]+')

ONLY_WORDS = re.compile(r'[.,!?]|(?:\'[a-z]*)') # for BOW

'''
- - - - -
Objects
- - - - -
'''

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

'''
- - - - -
Functions for .srt encoding
- - - - -
'''

def encoding_detector(file_path):
    '''
    This function takes file path as an argument,
    reads first 1000 bytes and of the file and
    retirns it's encoding as a string
    
    Requirements:
    - chardet module
    '''    
    # read the first 1000 bytes of the file
    with open(file_path, 'rb') as file:
        raw_data = file.read(1000)       
    # detect the encoding of the file
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.close()
    return encoding

def folder_to_utf(folder_path):
    '''
    This function takes folder path as anargument, gets all .srt files, encodes them to utf-8
    and puts them into /utf-8 subfolder
    
    Requirements:
    - os module
    - codecs module
    - encoding_detector function
    '''
    # create a utf-8 subfolder
    os.makedirs(os.path.join(folder_path, 'utf-8'), exist_ok=True)
    
    # loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.srt'):
            
            # define the file paths
            file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, 'utf-8', filename)

            # open the file and read its contents
            with codecs.open(file_path, 'r', encoding=encoding_detector(file_path), errors='replace') as file:
                contents = file.read()
                file.close()

            # write the contents to a new file with UTF-8 encoding
            with codecs.open(new_file_path, 'w', encoding='UTF-8', errors='replace') as new_file:
                new_file.write(contents)
                new_file.close()
                
'''
- - - - -
Functions for extracting text from .srt files
'''

def srt_raw_text(file_path):
    '''
    This function takes file path to .srt file as an argument
    and uses pysrt library (https://github.com/byroot/pysrt)
    to get text without timestamps
    
    Requirements:
    - pysrt module
    '''
    try:
        subs = pysrt.open(file_path)
        return subs.text
    except:
        return np.NaN

def srt_full_subs(file_path):
    '''
    This function takes file path to srt file as an argument
    and get all the text data "as is", including timestamps
    '''
    try:
        with open(file_path) as file:
            full_text = file.read()
            file.close()
        return full_text
    except:
        return np.NaN
    
'''
- - - - -
Functions for text preprocessing
- - - - -
'''

def re_clean_subs(subs):
    '''
    This function takes text as an argument and processes it using regular expressions
    Requirements:
    - re module
    - global variables:
    - - HTML = re.compile(r'<.*?>')
    - - TAG = re.compile(r'{.*?}')
    - - COMMENTS = re.compile(r'[\(\[][A-Z ]+[\)\]]')
    - - LETTERS = re.compile(r'[^a-zA-Z\'.,!? ]')
    - - SPACES = re.compile(r'([ ])\1+')
    - - DOTS = re.compile(r'[\.]+')
    '''
    txt = re.sub(HTML, ' ', subs) # html to space
    txt = re.sub(TAG, ' ', txt) # tags to space
    txt = re.sub(COMMENTS, ' ', txt) # commentaries to space
    txt = re.sub(LETTERS, ' ', txt) # non-char to space
    txt = re.sub(SPACES, r'\1', txt) # leading spaces to one space
    txt = re.sub(DOTS, r'.', txt)  # ellipsis to dot
    txt = txt.encode('ascii', 'ignore').decode() # clear non-ascii symbols   
    txt = ".".join(txt.lower().split('.')[1:-1]) # delete the first and the last subtitle (ads)
    return txt

def text_preprocess_lem(text):
    '''
    This function takes text as an argument and preprocess it using
    re_clean_subs function, nltk tokenizer, stop words list and lemmatizer
    Requirements:
    - nltk.word_tokenize
    - re_clean_subs function
    - variables:
    - - lemmatizer = nltk.stem.WordNetLemmatizer()
    - - stop_words = nltk.corpus.stopwords.words('english')
    '''
    text = re_clean_subs(text) # clean text using RE
    tokens = word_tokenize(text) # tokenisation
    text = [word for word in tokens if word not in stop_words] # stop words removal
    text = [lemmatizer.lemmatize(word) for word in text] # lemmatising tokens
    text = " ".join(text) # making text from the list
    return text

'''
- - - - -
Report function
- - - - -
'''

def report(y_test, y_pred):
    '''
    This function takes y_test and y_pred, prints
    subset accuracy score, confusion matrix heatmap and classification report
    Requirements:
    - sklearn.metrics:
    - - accuracy_score
    - - confusion_matrix
    - - classification_report
    - seaborn as sns
    '''
    print('Subset accuracy is:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:')
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=['A2', 'B1', 'B2', 'C1']),
                             index=['true A2', 'true B1', 'true B2', 'true C1'],
                             columns=['false A2', 'false B1', 'false B2', 'false C1']),
                annot=True)
    plt.show()
    print(classification_report(y_test, y_pred))