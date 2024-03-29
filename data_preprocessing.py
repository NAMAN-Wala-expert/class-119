#Text Data Preprocessing Lib
import nltk 
nltk.download('punkt')
from nltk.stem import PorterStemmer
stmr = PorterStemmer()
import json
import pickle
import numpy as np 
words = []
classes =[]
word_tag_list =[]
igwd = ['?','!',',','.',"'s","'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)



# function for appending stem words
def gt_stm_wrd (words,igwd) :
    stemwords = []
    for word in words :
        if word not in igwd :
            w = stmr.stem(word.lower())
            stemwords.append(w)
    return stemwords
        

for intent in intents ['intents']:
    # Add all words of patterns to list
    for pattern in intent['patterns'] :
        pattern_word =nltk.word_tokenize(pattern)
        words.extent(pattern_word)
        word_tag_list.append((pattern_word,intent['tag']))
    if intent['tag'] not in classes :
        classes.append(intent['tag'])
        stemwords = gt_stm_wrd(words,igwd)
# Add all tags to the classes list
print(stemwords)
print(word_tag_list)
print(classes)
     

        

#Create word corpus for chatbot
def create_bot_corpus(stemwords,classes):
    stemwords = sorted(list(set(stemwords)))
    classes = sorted(list(set(classes)))
    pickle.dump(stemwords,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    return stemwords,classes

stemwords,classes = create_bot_corpus(stemwords,classes)
print(stemwords)
print(classes)

train_data = []
no_of_tags = len(classes)
labels = [0]*no_of_tags


for word_tags in word_tag_list :
    bag_of_words =[]
    pattern_words = word_tags[0]
    for word in pattern_words:
        index = pattern_words.index(word)
        word = stmr.stem(word.lower())
        pattern_words[index]=word
    for word in stemwords :
        if word in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    print(bag_of_words)
    labels_encoding = list(labels)
    tag = word_tags[1]
    tag_index = classes.index(tag)
    labels_encoding[tag_index]=1
    train_data.append([bag_of_words,labels_encoding])

def preprocess_train_data(train_data) :
    train_data = np.array(train_data,ttype = object)
    train_x = list(train_data[:,0])
    train_y = list(train_data[:,1])
    print(train_x[0])
    print(train_y[0])
    return train_x,train_y
train_x,train_y = preprocess_train_data(train_data)