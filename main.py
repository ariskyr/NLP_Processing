from dataclasses import dataclass
import numpy as np
import re

#variables
Path = "DeliciousMIL\Data\\"
vocabs = {}

"""
Make a vocabs dictionary that looks like:
{0: ('word1', 0),
 1: ('word2', 1), etc
}
"""
with open((Path+str("vocabs.txt")), "r") as vocabsFile:
    for line in vocabsFile:
        (val,key) = line.split(",")
        vocabs[int(key)] = val,int(key)

#initialize trainData dictionary
sentenceDict = {}
numTrainDataLines = sum(1 for line in open((Path+str("train-data.dat"))))
trainData = {"Line " +str(List1): sentenceDict for List1 in range(1,numTrainDataLines+1)}

TrainDataLine = 0
with open((Path+str("train-data.dat")), "r") as trainDataFile:
    for line in trainDataFile:
        TrainDataLine += 1
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
        trainData["Line " +str(TrainDataLine)] = sentenceDict


x=0


