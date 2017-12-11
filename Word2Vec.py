import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import gensim, logging
import pickle
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def loadData(filename):
    input = open(filename)
    try:
        allFile = input.readlines()
    finally:
        input.close()
    return allFile

def getVocab():
    filenames = ["train_x.txt", "dev_x.txt", "test_x.txt"]
    dict = []
    for filename in filenames:
        fileContent = loadData(filename)

        for line in fileContent:
            eachShotText = []
            for word in line.strip('\n').split(' '):
                eachShotText.append(word)
            dict.append(eachShotText)
            #print(eachShotText)
    total_words = len(dict)
    print("total_words: ", total_words)

    return dict

def create(model =None):
    gensim_dist = Dictionary()
    #gensim_dist.doc2bow(p_model)
    gensim_dist.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx={v: k + 1 for k, v in gensim_dist.items()}
    print(w2indx)
    w2vec = {word: model[word] for word in w2indx.keys()}
    return w2indx, w2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = getVocab()
print(sentences)
model = Word2Vec(sentences,size=300,min_count=5,window=5)
model_name = "word.model"
model.save(model_name)

index_dict, word_vectors = create(model=model)
print(word_vectors)
print(index_dict)

pkl_name = "dictionary.pkl"
output = open(pkl_name,'wb')
pickle.dump(index_dict, output)
pickle.dump(word_vectors,output)
output.close()
# dict = getVocab()
# model= gensim.models.Word2Vec(dict, size=200, workers=4)
# #print(model.wv['computer'])
# mydict = Mydict(200,4,model)
# sentence = "a must see for all sides of the political spectrum"
# word2Vector = mydict.getVector(sentence)
# for vec in word2Vector:
#      print(vec)





