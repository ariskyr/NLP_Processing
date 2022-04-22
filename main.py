import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import Preprocess as pp

#########
#for the full preprocessing:
#########
#rawSplitSet,centeredSplitSet,NormalizedSplitSet = pp.preprocessData()

#########
# if you dont wanna wait for the preprocessing of data
#########
yo1 = pd.read_pickle("SavedArrays\\rawSplitSet.pkl")
yo2 = pd.read_pickle("SavedArrays\\centeredSplitSet.pkl")
yo3 = pd.read_pickle("SavedArrays\\normalizedSplitSet.pkl")

test1 = yo1['X_train'][4]
test2 = yo1['Y_train'][4]

x= 0