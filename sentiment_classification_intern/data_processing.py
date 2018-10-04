import pandas as pd
import emoji
import string
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec

# READ DATA
def read_data(link):
    train_data = pd.read_csv(link, sep="### ", header=None, engine='python')
    drop_list = train_data.index[train_data[0] == 4].tolist() # list of data with label 4
    train_data = train_data.drop(train_data.index[drop_list]) # drop these rows
    train_data = train_data.reset_index(drop=True)
    return train_data


def sentence2tokens(sentence):
    tokenized = word_tokenize(sentence)
    return tokenized


def remove_character(sentence):
    other_removal = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '«', '“', '”', '…', '»', '】☛', '►►', '🏿', '➤',
                     '◄◄', '🏻', '•', '·', '►']
    punctuation = list(string.punctuation)
    removal_character = other_removal + list(emoji.UNICODE_EMOJI.keys()) + punctuation
    sentence_split = ""
    for word in sentence:
        word = ''.join(ch for ch in word if ch not in removal_character)
        sentence_split += word
    return sentence_split


def remove_longwords(sentence):
    for word in sentence:
        if len(word) > 14:
            sentence.remove(word)


def lower(sentence):
    new_sentence = ""
    for word in sentence:
        new_sentence += word.lower() if any(x.isupper() for x in word) else word
    return new_sentence


def sentence_cleaning(sentence):
    new_sentence = remove_character(sentence)
    remove_longwords(new_sentence)
    lower_sentence = lower(new_sentence)
    tokenized = sentence2tokens(lower_sentence)
    return tokenized


# CREATE BAG OF WORDS
def createBoW(sentences):
    bow = []

    for i in range(len(sentences)):
        for word in sentences[i]:
            bow.append(word.lower())

    bow = list(set(bow))  # remove duplicate
    return bow


# PADDING + CHOPPING CLEAN SENTENCES


#longest_length = 50


def padORchop(sentences, longest):
    for i in range(len(sentences)):
        if len(sentences[i]) < longest:
            for j in range(longest - len(sentences[i])):
                sentences[i] += '0'
        if len(sentences[i]) > longest:
            sentences[i] = sentences[i][:50]


# padORchop(clean_sentences, longest_length)


# model = Word2Vec(clean_sentences, min_count=1, window=2, size=300, workers=20)
#
# model.wv.save_word2vec_format('model.txt', binary=False)
#
# model.save('model.bin')
#
# new_model = Word2Vec.load('model.bin')
#
# print(new_model)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

