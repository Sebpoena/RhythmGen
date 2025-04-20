#this the real shit
!pip install hmmlearn
from google.colab import drive
import csv
import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib

drive.mount('/content/drive')

import csv

def convertValue(value):
  """Convert to int or float if possible; otherwise return 'B'."""
  if value == 'B':
    return value
  try:
    return int(value)
  except ValueError:
    return float(value)

def csvToList(fileName):
    """Read a CSV file and convert values to int, float, or 'B'."""
    dataList = []
    with open(fileName, newline='', encoding='utf-8') as csvFile:
        csvReader = csv.reader(csvFile)
        for rowList in csvReader:
            convertedRow = [convertValue(value) for value in rowList]
            dataList.append(convertedRow)

    return dataList


class Compiler:
  def __init__(self, rawPhrases):
    self.rawPhrases = rawPhrases
    x = sum([len(i) for i in self.rawPhrases])
    print(x)
    self.preProcessedPhrases = []
    self.durations = [0.5, 1.0, 0.25, 0.75, 0.125,
                      2.0, 3.0, 0.3333, 1.5, 4.0, 0.375,
                      0.1667, 0.0625, 0.0833, 0.1875,
                      0.875, 0.6667, 1.75, 3.5, 3.75]
    for i in self.rawPhrases:
      self.splitLongPhrases(i)
    print(f"the length of the preprocessed phrases is {len(self.preProcessedPhrases)}")
    self.processedPhrases = [i for i in self.preProcessedPhrases if not self.sortRepetitivePhrases(i)]
    print(f"the length of the processed phrases is {len(self.processedPhrases)}")
    self.durToIndex = {dur: i for i, dur in enumerate(self.durations)}
    self.indexToDur = {i: dur for dur, i in self.durToIndex.items()}
    self.encodedPhrases = [self.encodePhrase(i) for i in self.processedPhrases]
    self.lengths = [32 for i in self.processedPhrases]

  def splitLongPhrases(self, phrases, maxLength=36, overlap=24):
    """this will separate longer phrases into shorter overlapping chunks"""
    print(f"the length of the phrases before separation is {len(phrases)}")
    count = 0
    for phrase in phrases:
      if len(phrase) > maxLength:
        for i in range(0, len(phrase), overlap):
          chunk = phrase[i:i + maxLength]
          if len(chunk) < maxLength:
            break
          self.preProcessedPhrases.append(chunk)
          count += 1
      else:
        self.preProcessedPhrases.append(phrase)
        count += 1
    print(f"the length of the phrases after separation is {count}")

  def sortRepetitivePhrases(self, phrase, threshold=1):
    """this will return True if a phrase is largely comprised of one duration"""
    """it will also safeguard against the unexpected durations or rounding errors"""
    if len(phrase) <= 5:
      return False
    durationCount = {}
    for duration in phrase:
      durationCount[duration] = durationCount.get(duration, 0) + 1
      if duration not in self.durations:
        self.durations.append(duration)
    percentage = max(durationCount.values())/len(phrase)
    return percentage > threshold

  def encodePhrase(self, phrase):
    for i in phrase:
      if i in self.durations:
        continue
      else:
        print('error')
    return [self.durToIndex[dur] for dur in phrase]


K189 = csvToList('/content/drive/MyDrive/MusicXML/K189_R_B.csv')
K457 = csvToList('/content/drive/MyDrive/MusicXML/K457_R_B.csv')
K331 = csvToList('/content/drive/MyDrive/MusicXML/K331_R_B.csv')
K545 = csvToList('/content/drive/MyDrive/MusicXML/K545_R_B.csv')
K279 = csvToList('/content/drive/MyDrive/MusicXML/K279_R_B.csv')
K570 = csvToList('/content/drive/MyDrive/MusicXML/K570_R_B.csv')
K576 = csvToList('/content/drive/MyDrive/MusicXML/K576_R_B.csv')
K310 = csvToList('/content/drive/MyDrive/MusicXML/K310_R_B.csv')
Op10No1 = csvToList('/content/drive/MyDrive/MusicXML/Op10No1_R_B.csv')
Op49No2 = csvToList('/content/drive/MyDrive/MusicXML/Op49No2_R_B.csv')
Op13 = csvToList('/content/drive/MyDrive/MusicXML/Op13_R_B.csv')
Op2No1 = csvToList('/content/drive/MyDrive/MusicXML/Op2No1_R_B.csv')
Op2No3 = csvToList('/content/drive/MyDrive/MusicXML/Op2No3_R_B.csv')
Op81a = csvToList('/content/drive/MyDrive/MusicXML/Op81a_R_B.csv')
Op78 = csvToList('/content/drive/MyDrive/MusicXML/Op78_R_B.csv')
Op31No1 = csvToList('/content/drive/MyDrive/MusicXML/Op31No1_R_B.csv')
D748 = csvToList("/content/drive/My Drive/MusicXML/D748_R_B.csv")
D959 = csvToList("/content/drive/My Drive/MusicXML/D959_R_B.csv")
D960 = csvToList("/content/drive/My Drive/MusicXML/D960_R_B.csv")
BraOp1 = csvToList("/content/drive/My Drive/MusicXML/BraOp1_R_B.csv")
COp36 = csvToList("/content/drive/My Drive/MusicXML/COp36_R_B.csv")
COp40No1 = csvToList("/content/drive/My Drive/MusicXML/COp40No1_R_B.csv")
FCOp35 = csvToList("/content/drive/My Drive/MusicXML/FCOp35_R_B.csv")
HobXVI23 = csvToList("/content/drive/My Drive/MusicXML/HobXVI23_R_B.csv")
HobXVI27 = csvToList("/content/drive/My Drive/MusicXML/HobXVI27_R_B.csv")
HobXVI34 = csvToList("/content/drive/My Drive/MusicXML/HobXVI34_R_B.csv")
HobXVI52 = csvToList("/content/drive/My Drive/MusicXML/HobXVI52_R_B.csv")
QK387 = csvToList("/content/drive/My Drive/MusicXML/QK387_R_B.csv")
QK421 = csvToList("/content/drive/My Drive/MusicXML/QK421_R_B.csv")
QK428 = csvToList("/content/drive/My Drive/MusicXML/QK428_R_B.csv")
QK458 = csvToList("/content/drive/My Drive/MusicXML/QK458_R_B.csv")
QK464 = csvToList("/content/drive/My Drive/MusicXML/QK464_R_B.csv")
QK465 = csvToList("/content/drive/My Drive/MusicXML/QK465_R_B.csv")
QK590 = csvToList("/content/drive/My Drive/MusicXML/QK590_R_B.csv")
QOp18No4 = csvToList("/content/drive/My Drive/MusicXML/QOp18No4_R_B.csv")
QOp130 = csvToList("/content/drive/My Drive/MusicXML/QOp130_R_B.csv")
QOp131 = csvToList("/content/drive/My Drive/MusicXML/QOp131_R_B.csv")
QOp18No1 = csvToList("/content/drive/My Drive/MusicXML/QOp18No1_R_B.csv")
QOp18No2 = csvToList("/content/drive/My Drive/MusicXML/QOp18No2_R_B.csv")
QOp18No5 = csvToList("/content/drive/My Drive/MusicXML/QOp18No5_R_B.csv")
QOp74 = csvToList("/content/drive/My Drive/MusicXML/QOp74_R_B.csv")
QOp95 = csvToList("/content/drive/My Drive/MusicXML/QOp95_R_B.csv")
QOp127 = csvToList("/content/drive/My Drive/MusicXML/Op127_R_B.csv") #there is a mistake here, one that I do not have the patience to fix
QOp132 = csvToList("/content/drive/My Drive/MusicXML/QOp132_R_B.csv")
HobIII33 = csvToList("/content/drive/My Drive/MusicXML/HobIII33_R_B.csv")
HobIII34 = csvToList("/content/drive/My Drive/MusicXML/HobIII34_R_B.csv")
HobIII63 = csvToList("/content/drive/My Drive/MusicXML/HobIII63_R_B.csv")
D667 = csvToList("/content/drive/My Drive/MusicXML/D667_R_B.csv")
D810 = csvToList("/content/drive/My Drive/MusicXML/D810_R_B.csv")
QMenOp12 = csvToList("/content/drive/My Drive/MusicXML/QMenOp12_R_B.csv")
QMenOp13 = csvToList("/content/drive/My Drive/MusicXML/QMenOp13_R_B.csv")
QMenOp44No2 = csvToList("/content/drive/My Drive/MusicXML/QMenOp44No2_R_B.csv")
QMenOp80 = csvToList("/content/drive/My Drive/MusicXML/QMenOp80_R_B.csv")
QBocOp39 = csvToList("/content/drive/My Drive/MusicXML/QBocOp39_R_B.csv")
QDvoOp96 = csvToList("/content/drive/My Drive/MusicXML/QDvoOp96_R_B.csv")
QOp59No1 = csvToList("/content/drive/My Drive/MusicXML/QOp59No1_R_B.csv")
QOp59No2 = csvToList("/content/drive/My Drive/MusicXML/QOp59No2_R_B.csv")
QOp59No3 = csvToList("/content/drive/My Drive/MusicXML/QOp59No3_R_B.csv")
QOp18No3 = csvToList("/content/drive/My Drive/MusicXML/QOp18No3_R_B.csv")
QOp18No6 = csvToList("/content/drive/My Drive/MusicXML/QOp18No6_R_B.csv")
QBraOp51No1 = csvToList("/content/drive/My Drive/MusicXML/QBraOp51No1_R_B.csv")
QBraOp51No2 = csvToList("/content/drive/My Drive/MusicXML/QBraOp51No2_R_B.csv")

pieces = [K189, K457, K331, K545, K279, K570, K576, K310, Op10No1,
          Op49No2, Op13, Op2No1, Op2No3, Op81a, Op78, Op31No1, D959,
          D960, D748, BraOp1, COp36, COp40No1, FCOp35, HobXVI23, HobXVI27,
          HobXVI34, HobXVI52, QK387, QK421, QK428, QK458, QK464, QK465,
          QK590, QOp18No4, QOp130, QOp131, QOp18No1, QOp18No2, QOp18No5,
          QOp74, QOp95, QOp127, QOp132, HobIII33, HobIII34, HobIII63,
          D667, D810, QMenOp12, QMenOp13, QMenOp44No2, QMenOp80, QBocOp39,
          QDvoOp96, QOp59No1, QOp59No2, QOp59No3, QOp18No3, QOp18No6,
          QBraOp51No1, QBraOp51No2]
comp1 = Compiler(pieces)

print(comp1.durations)

preFormatting = []
for i in comp1.encodedPhrases:
  preFormatting.extend(i)
formattedData = np.array(preFormatting, dtype = int).reshape(-1, 1)
print(formattedData)
formattedData = np.array(comp1.encodedPhrases, dtype = int)

model = hmm.MultinomialHMM(n_components=40, n_iter=100)
model.n_features = int(len(comp1.durations)/2)
model.fit(formattedData)

print(model.emissionprob_)
print(model.emissionprob_.shape)

def generate(model, length=16):
  model.n_trials = 32
  generated, _ = model.sample(n_samples = length)
  rhythm = generated.flatten().tolist()
  print([val for val in rhythm])
  return [comp1.indexToDur[i] for i in rhythm]

for i in range(30):
  print(generate(model, 1))

