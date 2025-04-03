!pip install hmmlearn
from hmmlearn import hmm
from google.colab import drive
import csv
import numpy as np
import pandas as pd
import joblib

drive.mount('/content/drive')

modelPath = '/content/drive/My Drive/MusicXML/m1.pkl' 

model = joblib.load(modelPath)

durations = [0.5, 1.0, 0.25, 0.75, 0.125, 2.0, 3.0, 0.3333, 1.5, 
             4.0, 0.375, 0.1667, 0.0625, 0.0833, 0.1875, 0.875, 
             0.6667, 1.75, 3.5, 3.75, 0.2, 0.3, 0.0417, 0.0354, 
             0.025, 0.0312, 0.0271, 0.4, 0.05, 0.4292, 0.0563, 
             0.0708, 0.1, 6.0, 0.0938, 1.3333]
durToIndex = {0.5: 0, 1.0: 1, 0.25: 2, 0.75: 3, 0.125: 4, 2.0: 5, 
              3.0: 6, 0.3333: 7, 1.5: 8, 4.0: 9, 0.375: 10, 
              0.1667: 11, 0.0625: 12, 0.0833: 13, 0.1875: 14, 
              0.875: 15, 0.6667: 16, 1.75: 17, 3.5: 18, 3.75: 19, 
              0.2: 20, 0.3: 21, 0.0417: 22, 0.0354: 23, 0.025: 24, 
              0.0312: 25, 0.0271: 26, 0.4: 27, 0.05: 28, 0.4292: 29, 
              0.0563: 30, 0.0708: 31, 0.1: 32, 6.0: 33, 0.0938: 34, 
              1.3333: 35}
indexToDur = {0: 0.5, 1: 1.0, 2: 0.25, 3: 0.75, 4: 0.125, 5: 2.0, 
              6: 3.0, 7: 0.3333, 8: 1.5, 9: 4.0, 10: 0.375, 11: 0.1667, 
              12: 0.0625, 13: 0.0833, 14: 0.1875, 15: 0.875, 16: 0.6667, 
              17: 1.75, 18: 3.5, 19: 3.75, 20: 0.2, 21: 0.3, 22: 0.0417, 
              23: 0.0354, 24: 0.025, 25: 0.0312, 26: 0.0271, 27: 0.4, 
              28: 0.05, 29: 0.4292, 30: 0.0563, 31: 0.0708, 32: 0.1, 
              33: 6.0, 34: 0.0938, 35: 1.3333}


def snapToNearestToken(value, tokens):
  return min(tokens, key=lambda token: abs(token - value))

def generate(model, length=16):
  generated, _ = model.sample(length)
  rhythm = generated.flatten().tolist()
  print([val for val in rhythm])
  return [indexToDur[snapToNearestToken(i, indexToDur)] for i in rhythm]

def listGeneratedRhythms(lengthList, lengthPhrases = 32):
  newGeneratedRhythms = []
  for i in range(lengthList):
    x = generate(model, lengthPhrases)
    newGeneratedRhythms.append(x)
  return newGeneratedRhythms

def encodeDecode(phrase, encoder):
  encoded = []
  for i in phrase:
    if i in encoder:
      encoded.append(encoder[i])
    else:
      continue
  return encoded

def continuePhrase(model, prompt, nSteps):
  """will generate based on a prompt given"""
  states = model.predict(prompt)
  lastState = states[-1]
  generated = prompt.flatten().tolist()
  for _ in range(nSteps):
    nextStateProbs = model.transmat_[lastState]
    nextState = np.random.choice(len(nextStateProbs), p=nextStateProbs)
    newObs = model.means_[nextState][0]
    snapped = snapToNearestToken(newObs, indexToDur)
    generated.append(snapped)
    lastState = nextState
  return generated

def singlePromptTest(model, prompt, nSteps, nPhrases = 100):
  formatted = np.array(encodeDecode(prompt, durToIndex)).reshape(-1, 1)
  output = []
  for i in range(nSteps):
    phrase = encodeDecode(continuePhrase(model, formatted, nSteps), indexToDur)
    output.append(phrase)
  return output

prompt = [0.5, 2.0, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5]

test = singlePromptTest(model, prompt, 15)

filePath = "/content/drive/My Drive/MusicXML/Test100_Same15.csv"

with open(filePath, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test)

def checkCSV(location):
  df = pd.read_csv(location, header=None)
  print(df.head())

checkCSV(filePath)


"""
phrase = [0.5, 2.0, 0.125, 0.125, 0.125, 0.125, 0.5, 1.0]
tokenised = encodeDecode(phrase, durToIndex)
reshaped = np.array(tokenised).reshape(-1, 1)

promptList = continuePhrase(model, reshaped, 12)
decoded = encodeDecode(promptList, indexToDur)
print(decoded)

genList = listGeneratedRhythms(15)
print(genList)
"""
