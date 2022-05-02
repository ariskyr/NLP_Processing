import pandas as pd
import numpy as np
import bz2
import pickle
import re
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#variables
PATH = "DeliciousMIL\\Data\\"

"""
Make a vocabs and labels dictionary that looks like:
{0: ('word1', 0),
 1: ('word2', 1), etc
}
"""
def importTxt(filename):
    Dict = {}
    with open((PATH+str(filename)), "r") as file:
        for line in file:
            (val,key) = line.split(",")
            Dict[int(key)] = val,int(key)
    output = list(Dict.values())
    return output

"""
import test and train datasets into dictionary:
{"Line 1": "Sentence 1": [1, 2, 3, 4], "Sentence 2": [3,4,1,2]
 "Line 2": "Sentence 1": [1, 2, 3, 4], "Sentence 2": [3,4,1,2]
}
"""
def importSet(filename):
    #initialize trainData dictionary
    sentenceDict = {}
    numDataLines = sum(1 for line in open((PATH+str(filename))))
    Dataset = {"Line " +str(List1): sentenceDict for List1 in range(1,numDataLines+1)}

    DataLine = 0
    with open((PATH+str(filename)), "r") as DataFile:
        for line in DataFile:
            DataLine += 1
            #line into list separated by whitespacce
            lineContent = line.split()
            lineCount = 0
            sentenceCount = 0
            wordNum = 0
            for s in lineContent:
                #for the first <> in each line (meaning sentences)
                if("<" in s and (lineCount==0)):
                    # ex. get '<2>' and return the integer value 2
                    sentenceNum = [int(s) for s in re.findall(r'\b\d+\b', s)][0]
                    sentenceDict = {"Sentence " +str(List2): [] for List2 in range(1,sentenceNum+1)}
                    lineCount += 1
                #for the rest of <> (meaning words per sentence)
                elif("<" in s and (wordNum==0)):
                    wordNum = [int(s) for s in re.findall(r'\b\d+\b', s)][0]
                    sentenceCount += 1
                else:
                    sentenceDict["Sentence " +str(sentenceCount)].append(int(s))
                    wordNum = wordNum -1
            Dataset["Line " +str(DataLine)] = sentenceDict
    return Dataset    

"""
implement bag of words model
{"Line 1": [0 0 0 ... 1...0] <- size: 8520 the same as vocabs.txt
 "Line 2": [0 0 0 ... 1...0] <- size: 8520 the same as vocabs.txt
 ...
 "Line (size of data): [...]
}
"""
def calculateBOW(dataset, vocabs):
    #get all words [0,1,2,...,8520]
    words = []
    for index,word in enumerate(vocabs):
        words.append(word[1])

    outputArray = []
    for line in dataset:
        vector = []
        #for each document, get all words from all sentences
        sentences = sum(dataset[line].values(), [])
        #count occurences of words and append to output
        for word in words:
            vector.append(sentences.count(word))
        outputArray.append(vector)
    return outputArray

#import all data
def importData():
    vocabs = importTxt("vocabs.txt")
    labels = importTxt("labels.txt")
    trainSet = importSet("train-data.dat")
    testSet = importSet("test-data.dat")
    trainLabels = np.loadtxt(PATH+str("train-label.dat")).astype(int)
    testLabels = np.loadtxt(PATH+str("test-label.dat")).astype(int)
    return vocabs,labels,trainSet,testSet,trainLabels,testLabels

#merge the 2 datasets
def mergeSets(datasetBOW, labels):
    OutputSet = pd.DataFrame(list(zip(datasetBOW, labels)), columns=['X_train', 'Y_train'])
    return OutputSet

#concat the datasets and convert to numpy arrays
def createNPArrays(TrainSet,TestSet):
    MergedSet = pd.concat([TrainSet,TestSet])
    X_train = np.array(MergedSet['X_train'].to_list())
    Y_train = np.array(MergedSet['Y_train'].to_list())
    return X_train,Y_train

#Centering
def centerData(X_train):
    meanSet = np.mean(X_train,axis=1).reshape(-1,1)
    X_trainCentered = X_train - meanSet
    return X_trainCentered

#Normalization between [0,1]
def normalizeData(X_train):
    scaler = MinMaxScaler()
    X_trainNormalized = scaler.fit_transform(X_train)
    return X_trainNormalized

#Standardization with mean=0 and std var=1
def standardizeData(X_train):
    scaler = StandardScaler()
    X_trainStandardized = scaler.fit_transform(X_train)
    return X_trainStandardized

#split dataframe into k chunks, we will then use each chunk as a testing set
def makeDataframe(X_train, Y_train):
    Dataset = pd.DataFrame()
    Dataset["X_train"] = pd.Series(list(X_train))
    Dataset["Y_train"] = pd.Series(list(Y_train))
    return Dataset

#pickle and compress
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        pickle.dump(data,f)
#decompress
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def preprocessData():
    #import data
    vocabs,labels,trainSet,testSet,trainLabels,testLabels = importData()

    #calculate Bag of Words model
    #this may take 2-3 minutes
    trainBOW = calculateBOW(trainSet,vocabs)
    testBOW = calculateBOW(testSet,vocabs)

    #merge
    trainSet = mergeSets(trainBOW,trainLabels)
    testSet = mergeSets(testBOW,testLabels)
    
    #preprocessed data
    X_train, Y_train = createNPArrays(trainSet,testSet)
    X_trainCentered = centerData(X_train)
    X_trainNormalized = normalizeData(X_train)
    X_trainStandardized = standardizeData(X_train)
    X_trainCN = normalizeData(X_trainCentered)
    
    #5 fold CV split
    rawSplitSet = makeDataframe(X_train,Y_train)
    centeredSplitSet = makeDataframe(X_trainCentered,Y_train)
    NormalizedSplitSet = makeDataframe(X_trainNormalized,Y_train)
    StandardizedSplitSet = makeDataframe(X_trainStandardized,Y_train)
    CNSplitSet = makeDataframe(X_trainCN, Y_train)

    #compress and save files
    compressed_pickle('SavedArrays\\rawSplitSet', rawSplitSet)
    compressed_pickle('SavedArrays\\centeredSplitSet', centeredSplitSet)
    compressed_pickle('SavedArrays\\normalizedSplitSet', NormalizedSplitSet)
    compressed_pickle('SavedArrays\\standardizedSplitSet', StandardizedSplitSet)
    compressed_pickle('SavedArrays\\CNSplitsSet', CNSplitSet)

    return rawSplitSet,centeredSplitSet,NormalizedSplitSet, StandardizedSplitSet, CNSplitSet

