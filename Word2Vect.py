import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词


import gensim, logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Mydict(object):
    def __init__(self, nsize, workers,model):
        self.nsize = nsize
        self.workers = workers
        self.model = model

    def getVector(self, context):

        vec = np.zeros(self.nsize)
        count = 0
        for word in context.strip('\n').split(' '):
            if word in self.model:
                vec += self.model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        return vec

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

dict = getVocab()
model= gensim.models.Word2Vec(dict, size=200, workers=4)
#print(model.wv['computer'])
mydict = Mydict(200,4,model)
sentence = "a must see for all sides of the political spectrum"
word2Vector = mydict.getVector(sentence)
for vec in word2Vector:
     print(vec)





