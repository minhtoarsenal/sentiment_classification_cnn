import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import emoji
import string
from nltk import sent_tokenize
from nltk import word_tokenize
#from keras.preprocessing.sequence import pad_sequences

# READ DATA
train_data = pd.read_csv('Data/train.txt', sep="### ", header=None, engine='python')
drop_list = train_data.index[train_data[0] == 4].tolist() # list of data with label 4
train_data = train_data.drop(train_data.index[drop_list]) # drop these rows
train_data = train_data.reset_index(drop=True)


# EACH COMMENT TO SENTENCES
all_sentences = []
for i in range(len(train_data)):
    tokenized = sent_tokenize(train_data[1][i])
    for element in range(len(tokenized)):
        all_sentences.append(tokenized[element])


# characters that needs to be removed
other_removal = ['0','1','2','3','4','5','6','7','8','9','«','“','”','…','»','】☛','►►','🏿','➤','◄◄','🏻','•','·','►']

punctuation = list(string.punctuation)

removal_character = other_removal + list(emoji.UNICODE_EMOJI.keys()) + punctuation


# CREATE BAG OF WORDS
boW = []
clean_sentences = []


def remove_character(sentence):
    sentence_split = ""
    for word in sentence:
        for ch in word:
            word = ''.join(ch for ch in word if ch not in removal_character)
            sentence_split += word
    return sentence_split


def sentence2word(sentence):
    tokenized = word_tokenize(sentence)
    return tokenized


for i in range(len(all_sentences)):
    clean_sentence = remove_character(all_sentences[i])
    tokenized = word_tokenize(clean_sentence)
    clean_sentences.append(tokenized)   # all clean sentences
    for word in tokenized:
        boW.append(word.lower())  # BoW


print('number of words in bag of words is: %d' % len(boW))

# remove too long words and stop words
long_words = []
for ch in boW:
    if len(ch) > 14:
        long_words.append(ch)

vnese_stop_words = ['bị','bởi','cả','các','cái','cần','càng','chỉ','chiếc','cho','chứ','chưa','chuyện','có','có_thể','cứ',
                    'của','cùng','cũng','đã','đang','đây','để','đến_nỗi','đều','điều','do','đó','được','dưới','gì',
                    'khi','không','là','lại','lên','lúc','mà','mỗi','một_cách','này','nên','nếu','ngay','nhiều','như',
                    'nhưng','những','nơi','nữa','phải','qua','ra','rằng','rất','rồi','sau','sẽ','so','sự','tại','theo',
                    'thì','trên','trước','từ','từng','và','vẫn','vào','vậy','vì','việc','với','vừa']


boW_ = [word for word in boW if word not in vnese_stop_words not in long_words]
print('number of words in bag of words now is: %d' % len(boW_))


word2int = {}
int2word = {}


for i, word in enumerate(boW_):
    word2int[word] = i
    int2word[i] = word

word2int.update({'0': 0})  # for padding sentences with 0
int2word.update({'0': '0'})  # for padding sentences with 0


# MODIFY AND PADDING CLEAN SENTENCES
for sentence in clean_sentences:
    for word in sentence:
        if len(word) > 14:
            sentence.remove(word)


def longest_sentence(sentences):
    position = 0
    longest = 0
    for num, sentence in enumerate(sentences):
        if len(sentences[num]) > longest:
            longest = len(sentences[num])
            position = num

    return position, longest


def longest_word(sentences):
    longest_word_length = 0
    long_word = ""
    for sentence in sentences:
        for word in sentence:
            if len(word) > longest_word_length:
                longest_word_length = len(word)
                long_word = word
    return long_word

print('the longest word is %s' %(longest_word(clean_sentences)))
pos, longest_num = longest_sentence(clean_sentences)
print('pos is %d, longest_num is %d' % longest_sentence(clean_sentences))
#print(clean_sentences[2135])

def pad(sentences, longest):
    for i in range(len(sentences)):
        if len(sentences[i]) < longest:
            for j in range(longest - len(sentences[i])):
                sentences[i] += '0'


pad(clean_sentences, longest_num)

#print(clean_sentences[2])
# print(clean_sentences)






