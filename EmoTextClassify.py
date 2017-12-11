from __future__ import print_function

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,MaxPool1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import *
import keras
import os
import sys
import numpy as np
import codecs
import os
import pickle


# BASE_DIR = ''
# GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
VOCAB_DIM= 300  # vector dim
MAXLEN = 55  # text remained max length
BATCH_SIZE = 256
N_EPOCH = 5
INPUT_LENGTH = 55

def WriteToFile(data, filename):
    f = open(filename, 'w')
    predicted_label = []
    for line in data:
        line = list(line)
        index = line.index(max(line)) + 1
        predicted_label.append(index)
        f.write(str(index) + '\n', )
        # print("index:" , index)
    print("predicted label: ", predicted_label)
    f.close()


def loadData(filename):
    fopen = open(filename)
    data = []
    for line in fopen.readlines():
        data.append(line.strip('\n'))
    # print(data)
    return data


def getDataSentence(filename):
    data = loadData(filename)
    dataX = []
    for line in data:
        linedata = []
        line = line.strip('\n').split(' ')
        for word in line:
            linedata.append(word)
        dataX.append(linedata)

    print("dataX:", len(dataX))
    return np.array(dataX)


def getDataLabel(filename):
    data = loadData(filename)
    dataY = []
    for line in data:
        dataY.append(int(line) - 1)
    # print(dataY)
    print("dataY:", len(dataY))
    return np.array(dataY)





# text to index
def text_to_index_array(p_new_dic, p_sen):
    new_sentence = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])
            except:
                new_sen.append(0)
        new_sentence.append(new_sen)
    return np.array(new_sentence)

#LSTM train
def train_lstm(p_n_symbols, p_emdedding_weights, p_X_train, p_y_train, p_X_test, p_y_test, p_X_predict):
    print("模型开始创建...")
    model = Sequential()
    model.add(Embedding(
        output_dim=VOCAB_DIM,
        input_dim=p_n_symbols,
        #mask_zero=True,
        weights=[p_emdedding_weights],
        input_length=INPUT_LENGTH
    ))

    # model.add(Convolution1D(128, 3, padding='same', strides=1))
    # model.add(Activation('relu'))
    # # model.add(MaxPool1D(pool_size=4))
    # model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))

    # model.add(GRU(
    #     output_dim=64,
    #     activation='sigmoid',
    #     inner_activation='hard_sigmoid'
    # ))
    # model.add(Dropout(0.2))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    print("模型开始编译...")

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy']
                  )

    print("开始训练模型...")

    model.fit(p_X_train, p_y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCH,
              validation_data=(p_X_test, p_y_test))

    print("模型评估...")
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=BATCH_SIZE)
    print("Test score: ", score)
    print('Test accuracy:', acc)
    model.save("LSTMmodel.h5")
    predicted = model.predict(p_X_predict)
    WriteToFile(predicted, "MF1733056.txt")
    #print(predicted)


if __name__=="__main__":
    f = open("dictionary.pkl", 'rb')
    index_dict = pickle.load(f)  # 索引词典
    # print("index_dict:", index_dict)
    word_vectors = pickle.load(f)  # 词向量
    # print("word_vect:", word_vectors)
    new_dic = index_dict


    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, 300))

    for w, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[w]
    print("embedding_weights length: ", embedding_weights.shape)
    train_X = getDataSentence("train_x.txt")
    #print("trainX:", train_X.shape)
    dev_X = getDataSentence("dev_x.txt")
    test_X = getDataSentence("test_x.txt")
    train_y = getDataLabel("train_y.txt")
    dev_y = getDataLabel("dev_y.txt")

    X_train = text_to_index_array(new_dic, train_X)
    X_dev = text_to_index_array(new_dic, dev_X)
    X_test = text_to_index_array(new_dic, test_X)

    print("训练集shape: ", X_train.shape)
    print("测试机shape: ", X_dev.shape)

    y_train = keras.utils.to_categorical(train_y, num_classes=5)
    y_dev = keras.utils.to_categorical(dev_y, num_classes=5)
    #print(y_train)
    X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
    X_dev = sequence.pad_sequences(X_dev, maxlen=MAXLEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)
    print("训练集shape: ", X_train.shape)
    print("测试机shape: ", X_dev.shape)

    train_lstm(n_symbols, embedding_weights, X_train, y_train, X_dev, y_dev, X_test)
