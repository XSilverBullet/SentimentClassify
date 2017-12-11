from __future__ import print_function

import os
import sys
import numpy as np
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, concatenate
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D,LSTM,GRU,Bidirectional
from keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras.models import Sequential



BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = 55
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300


def utf_open(name):
    return codecs.open(name, "r", encoding="utf-8")


def get_data():
    train_x_file = utf_open(os.path.join(TEXT_DATA_DIR, "train_x.txt"))
    train_y_file = utf_open(os.path.join(TEXT_DATA_DIR, "train_y.txt"))
    train_texts = train_x_file.read().split("\n")
    train_labels = train_y_file.read().split("\n")
    # the final word is ''
    train_labels.pop()
    train_texts.pop()

    dev_x_file = utf_open(os.path.join(TEXT_DATA_DIR, "dev_x.txt"))
    dev_y_file = utf_open(os.path.join(TEXT_DATA_DIR, "dev_y.txt"))
    dev_texts = dev_x_file.read().split("\n")
    dev_labels = dev_y_file.read().split("\n")
    # the final word is ''
    dev_labels.pop()
    dev_texts.pop()

    test_x_file = utf_open(os.path.join(TEXT_DATA_DIR, "test_x.txt"))
    test_texts = test_x_file.read().split("\n")
    test_texts.pop()

    return train_texts, train_labels, dev_texts, dev_labels, test_texts


def main():
    print('Indexing word vectors.')

    embeddings_index = {}
    f = utf_open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing dataset')

    train_texts, train_labels, dev_texts, dev_labels, test_texts = get_data()

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_texts + dev_texts + test_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    dev_sequences = tokenizer.texts_to_sequences(dev_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    dev_data = pad_sequences(dev_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    train_labels = to_categorical(np.asarray(train_labels, dtype=int) - 1)
    dev_labels = to_categorical(np.asarray(dev_labels, dtype=int) - 1)

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of train_label tensor:', train_labels.shape)

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Convolution1D(256, 3, padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(5, activation='softmax'))
    print('Training model.')

    # train a 1D convnet with global maxpooling
    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embed = embedding_layer(sequence_input)
    # embed = Dropout(0.2)(embed)
    #
    # # cnn1  kernel_size = 3
    # conv1_1 = Convolution1D(256, 3, padding='same')(embed)
    # bn1_1 = BatchNormalization()(conv1_1)
    # relu1_1 = Activation('relu')(bn1_1)
    # cnn1 = MaxPool1D(pool_size=4)(relu1_1)
    # # cnn2  kernel_size = 4
    # conv2_1 = Convolution1D(256, 4, padding='same')(embed)
    # bn2_1 = BatchNormalization()(conv2_1)
    # relu2_1 = Activation('relu')(bn2_1)
    # cnn2 = MaxPool1D(pool_size=4)(relu2_1)
    # # cnn3  kernel_size = 5
    # conv3_1 = Convolution1D(256, 5, padding='same')(embed)
    # bn3_1 = BatchNormalization()(conv3_1)
    # relu3_1 = Activation('relu')(bn3_1)
    # cnn3 = MaxPool1D(pool_size=4)(relu3_1)
    #
    # # concatenate
    # cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    # flat = Flatten()(cnn)
    # drop = Dropout(0.5)(flat)
    # fc = Dense(512)(drop)
    # bn = BatchNormalization()(fc)
    #
    # x = Dense(256, activation='relu')(bn)
    # preds = Dense(5, activation='softmax')(x)

    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # #main_input = Input(shape=(20,), dtype='float64')
    # embed = embedding_layer(sequence_input)
    # cnn = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    # cnn = MaxPool1D(pool_size=4)(cnn)
    # cnn = Flatten()(cnn)
    # cnn = Dense(256)(cnn)
    # rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
    # rnn = Dense(256)(rnn)
    # con = concatenate([cnn, rnn], axis=-1)
    # main_output = Dense(5, activation='softmax')(con)
    # model = Model(inputs=sequence_input, outputs=main_output)


    #model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    # model = load_model("cnn.h5")
    model.fit(train_data, train_labels,
              batch_size=256,
              epochs=5,
              validation_data=(dev_data, dev_labels))

    model.save("cnn.h5")

    predict = model.predict(test_data).argmax(axis=1) + 1
    predict = predict.astype("int")
    np.savetxt("MF1733056.txt", predict.reshape(-1, 1), fmt="%1s")


if __name__ == "__main__":
    main()

