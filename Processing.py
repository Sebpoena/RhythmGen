#this the real shit
!pip install hmmlearn
from google.colab import drive
import csv
import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib

drive.mount('/content/drive')

def csvToList(filename):
  with open(filename, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
      new_row = []
      for x in row:
        x = x.strip()
        if x and x != "None":
          try:
            new_row.append(float(x))
          except ValueError:
            print(f"Skipping value: {x}")
      if new_row:
        data.append(new_row)
  return data

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


K189 = csvToList('/content/drive/MyDrive/MusicXML/K189.csv')
K457 = csvToList('/content/drive/MyDrive/MusicXML/K457.csv')
K331 = csvToList('/content/drive/MyDrive/MusicXML/K331.csv')
K545 = csvToList('/content/drive/MyDrive/MusicXML/K545.csv')
K279 = csvToList('/content/drive/MyDrive/MusicXML/K279.csv')
K570 = csvToList('/content/drive/MyDrive/MusicXML/K570.csv')
K576 = csvToList('/content/drive/MyDrive/MusicXML/K576.csv')
K310 = csvToList('/content/drive/MyDrive/MusicXML/K310.csv')
Op10No1 = csvToList('/content/drive/MyDrive/MusicXML/Op10No1.csv')
Op49No2 = csvToList('/content/drive/MyDrive/MusicXML/Op49No2.csv')
Op13 = csvToList('/content/drive/MyDrive/MusicXML/Op13.csv')
Op2No1 = csvToList('/content/drive/MyDrive/MusicXML/Op2No1.csv')
Op2No3 = csvToList('/content/drive/MyDrive/MusicXML/Op2No3.csv')
Op81a = csvToList('/content/drive/MyDrive/MusicXML/Op81a.csv')
Op78 = csvToList('/content/drive/MyDrive/MusicXML/Op78.csv')
Op31No1 = csvToList('/content/drive/MyDrive/MusicXML/Op31No1.csv')
D748 = csvToList("/content/drive/My Drive/MusicXML/D748.csv")
D959 = csvToList("/content/drive/My Drive/MusicXML/D959.csv")
D960 = csvToList("/content/drive/My Drive/MusicXML/D960.csv")
BraOp1 = csvToList("/content/drive/My Drive/MusicXML/BraOp1.csv")
COp36 = csvToList("/content/drive/My Drive/MusicXML/COp36.csv")
COp40No1 = csvToList("/content/drive/My Drive/MusicXML/COp40No1.csv")
FCOp35 = csvToList("/content/drive/My Drive/MusicXML/FCOp35.csv")
HobXVI23 = csvToList("/content/drive/My Drive/MusicXML/HobXVI23.csv")
HobXVI27 = csvToList("/content/drive/My Drive/MusicXML/HobXVI27.csv")
HobXVI34 = csvToList("/content/drive/My Drive/MusicXML/HobXVI34.csv")
HobXVI52 = csvToList("/content/drive/My Drive/MusicXML/HobXVI52.csv")
QK387 = csvToList("/content/drive/My Drive/MusicXML/QK387.csv")
QK421 = csvToList("/content/drive/My Drive/MusicXML/QK421.csv")
QK428 = csvToList("/content/drive/My Drive/MusicXML/QK428.csv")
QK458 = csvToList("/content/drive/My Drive/MusicXML/QK458.csv")
QK464 = csvToList("/content/drive/My Drive/MusicXML/QK464.csv")
QK465 = csvToList("/content/drive/My Drive/MusicXML/QK465.csv")
QK590 = csvToList("/content/drive/My Drive/MusicXML/QK590.csv")
QOp18No4 = csvToList("/content/drive/My Drive/MusicXML/QOp18No4.csv")
QOp130 = csvToList("/content/drive/My Drive/MusicXML/QOp130.csv")
QOp131 = csvToList("/content/drive/My Drive/MusicXML/QOp131.csv")
QOp18No1 = csvToList("/content/drive/My Drive/MusicXML/QOp18No1.csv")
QOp18No2 = csvToList("/content/drive/My Drive/MusicXML/QOp18No2.csv")
QOp18No5 = csvToList("/content/drive/My Drive/MusicXML/QOp18No5.csv")
QOp74 = csvToList("/content/drive/My Drive/MusicXML/QOp74.csv")
QOp95 = csvToList("/content/drive/My Drive/MusicXML/QOp95.csv")
QOp127 = csvToList("/content/drive/My Drive/MusicXML/QOp127.csv")
QOp132 = csvToList("/content/drive/My Drive/MusicXML/QOp132.csv")
HobIII33 = csvToList("/content/drive/My Drive/MusicXML/HobIII33.csv")
HobIII34 = csvToList("/content/drive/My Drive/MusicXML/HobIII34.csv")
HobIII63 = csvToList("/content/drive/My Drive/MusicXML/HobIII63.csv")
D667 = csvToList("/content/drive/My Drive/MusicXML/D667.csv")
D810 = csvToList("/content/drive/My Drive/MusicXML/D810.csv")
QMenOp12 = csvToList("/content/drive/My Drive/MusicXML/QMenOp12.csv")
QMenOp13 = csvToList("/content/drive/My Drive/MusicXML/QMenOp13.csv")
QMenOp44No2 = csvToList("/content/drive/My Drive/MusicXML/QMenOp44No2.csv")
QMenOp80 = csvToList("/content/drive/My Drive/MusicXML/QMenOp80.csv")
QBocOp39 = csvToList("/content/drive/My Drive/MusicXML/QBocOp39.csv")
QDvoOp96 = csvToList("/content/drive/My Drive/MusicXML/QDvoOp96.csv")
QOp59No1 = csvToList("/content/drive/My Drive/MusicXML/QOp59No1.csv")
QOp59No2 = csvToList("/content/drive/My Drive/MusicXML/QOp59No2.csv")
QOp59No3 = csvToList("/content/drive/My Drive/MusicXML/QOp59No3.csv")
QOp18No3 = csvToList("/content/drive/My Drive/MusicXML/QOp18No3.csv")
QOp18No6 = csvToList("/content/drive/My Drive/MusicXML/QOp18No6.csv")
QBraOp51No1 = csvToList("/content/drive/My Drive/MusicXML/QBraOp51No1.csv")
QBraOp51No2 = csvToList("/content/drive/My Drive/MusicXML/QBraOp51No2.csv")

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

formattedData = [np.array(phrase).reshape(-1, 1) for phrase in comp1.encodedPhrases]

n_states = 15 #reminder - play around with setting if not working
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=250)
model.fit(np.vstack(formattedData))

def snapToNearestToken(value, tokens):
  return min(tokens, key=lambda token: abs(token - value))

def generate(model, length=16):
  generated, _ = model.sample(length)
  rhythm = generated.flatten().tolist()
  print([val for val in rhythm])
  return [comp1.indexToDur[snapToNearestToken(i, comp1.durToIndex)] for i in rhythm]

#rhythmGen = generate(model, length=20)

def listGeneratedRhythms(lengthList, lengthPhrases = 32):
  newGeneratedRhythms = []
  for i in range(lengthList):
    x = generate(model, lengthPhrases)
    newGeneratedRhythms.append(x)
  return newGeneratedRhythms

rhythms = listGeneratedRhythms(50)

modelPath = "/content/drive/My Drive/MusicXML/m2.pkl"

joblib.dump(model, modelPath)

filePath = "/content/drive/My Drive/MusicXML/Results15_250_1e-2_32x50.csv"

with open(filePath, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rhythms)

def checkCSV(location):
  df = pd.read_csv(location, header=None)
  print(df.head())

checkCSV(filePath)
