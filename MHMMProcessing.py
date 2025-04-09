#new code to retrieve all information, including rests and barlines
from google.colab import drive
import music21
import csv
import pandas as pd

drive.mount('/content/drive')

K279 = music21.converter.parse('/content/drive/MyDrive/MusicXML/K279_2.mxl')

def translate(flatPart):
  translated = []
  for i in flatPart:
    if i == 'B':
      translated.append('B')
    elif isinstance(i, music21.note.Note):
      if i.quarterLength > 0:
        translated.append(round(float(i.quarterLength), 4))
      else:
        continue
    elif isinstance(i, music21.chord.Chord):
      if i.quarterLength > 0:
        translated.append(round(float(i.quarterLength), 4))
      else:
        continue
    elif isinstance(i, music21.note.Rest):
      if i.quarterLength > 0:
        translated.append(round(float(i.quarterLength), 4) * -1)
  return translated

def flattenAndTranslate(part):
  """retrieves the information of a part and makes it understandable for later"""
  flattened = []
  for bar in part.getElementsByClass('Measure'):
    voices = bar.getElementsByClass('Voice')
    if voices:
      flattened.extend(voices[0].notesAndRests)
    else:
      flattened.extend(bar.notesAndRests)
    flattened.extend('B')
  return translate(flattened)

def treatScoreSolo(piece):
  part = piece.parts[0]
  return flattenAndTranslate(part)

def treatScoreQuartet(piece):
  partList = piece.parts[:3:]
  returnList = []
  for part in partList:
    returnList.extend(flattenAndTranslate(part))
  return returnList

def separateIntoChunks(list, length = 32, overlap = 16):
  returnList = []
  for i in range(0, len(list), overlap):
    chunk = list[i:i + length]
    returnList.append(chunk)
  return returnList

x = flattenAndTranslate(K279.parts[0])
print(x)
print(len(x))
y = separateIntoChunks(x)
print(y)
print(len(y))
