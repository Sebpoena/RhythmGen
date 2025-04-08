#code for generating phrases of different lengths with natural endings
!pip install hmmlearn
from google.colab import drive
import csv
import numpy as np
from hmmlearn import hmm
import joblib

drive.mount('/content/drive')

modelPath = '/content/drive/My Drive/MusicXML/m1.pkl'
model = joblib.load(modelPath)

s = {7: 2444, 12: 2068, 13: 211, 3: 62, 0: 17, 14: 2}
m = {7: 1489, 12: 1483, 13: 120, 0: 11, 3: 32, 14: 2}
l = {7: 1579, 12: 1574, 13: 133, 3: 32, 0: 10}

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
  """finds the nearest real token to the model output"""
  return min(tokens, key=lambda token: abs(token - value))

def generate(model, length=16):
  """will generate phrases of a specified length with a specified model"""
  generated, _ = model.sample(length)
  rhythm = generated.flatten().tolist()
  print([val for val in rhythm])
  return [indexToDur[snapToNearestToken(i, indexToDur)] for i in rhythm]

def encodeDecode(phrase, encoder):
  """classic encoder to tokenise the data"""
  encoded = []
  for i in phrase:
    if i in encoder:
      encoded.append(encoder[i])
    else:
      continue
  return encoded

def decideEnding(dict):
  keys = list(dict.keys())
  values = list(dict.values())
  return random.choices(keys, weights = [i/sum(values) for i in values])[0]

def naturalEndPrompted(model, prompt, minLength, dict):
  """will generate natural phrase based on a prompt given"""
  switch = True
  formattedPrompt = np.array(encodeDecode(prompt, durToIndex)).reshape(-1, 1)
  states = model.predict(formattedPrompt)
  lastState = states[-1]
  generated = formattedPrompt.flatten().tolist()
  while switch:
    nextStateProbs = model.transmat_[lastState]
    nextState = np.random.choice(len(nextStateProbs), p=nextStateProbs)
    newObs = model.means_[nextState][0]
    snapped = snapToNearestToken(newObs, indexToDur)
    generated.append(snapped)
    lastState = nextState
    if len(generated) >= minLength and lastState == decideEnding(dict):
      switch = False
  return generated

def naturalEndUnprompted(model, minLength, dict):
  """will generate natural phrase without a prompt"""
  switch = True
  generated = generate(model, minLength - 2)
  formatted = np.array(encodeDecode(generated, durToIndex)).reshape(-1, 1)
  states = model.predict(formatted)
  lastState = states[-1]
  generated = formatted.flatten().tolist()
  while switch:
    nextStateProbs = model.transmat_[lastState]
    nextState = np.random.choice(len(nextStateProbs), p=nextStateProbs)
    newObs = model.means_[nextState][0]
    snapped = snapToNearestToken(newObs, indexToDur)
    generated.append(snapped)
    lastState = nextState
    if lastState == decideEnding(dict):
      switch = False
  return generated

