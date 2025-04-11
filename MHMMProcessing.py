#new code to retrieve all information, including rests and barlines
from google.colab import drive
import music21
import csv
import pandas as pd

drive.mount('/content/drive')

QK387 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK387_1.mxl")
QK421 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK421_1.mxl")
QK428 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK428_1.mxl")
QK458 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK458_1.mxl")
QK464 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK464_1.mxl")
QK465 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK465_1.mxl")
QK590 = music21.converter.parse("/content/drive/My Drive/MusicXML/QK590_1.mxl")
QOp18No4 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No4_1.mxl")
QOp130 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp130_1.mxl")
QOp131 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp131_1.mxl")

QOp18No1 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No1_1.mxl")
QOp18No2 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No2_1.mxl")
QOp18No5 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No5_1.mxl")
QOp74 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp74_1.mxl")
QOp95 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp95_1.mxl")
QOp127 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp127_1.mxl")
QOp132 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp132_1.mxl")

HobIII33 = music21.converter.parse("/content/drive/My Drive/MusicXML/HobIII33_1.mxl")
HobIII34 = music21.converter.parse("/content/drive/My Drive/MusicXML/HobIII34_1.mxl")
HobIII63 = music21.converter.parse("/content/drive/My Drive/MusicXML/HobIII63_1.mxl")
D667 = music21.converter.parse("/content/drive/My Drive/MusicXML/D667_1.mxl")
D810 = music21.converter.parse("/content/drive/My Drive/MusicXML/D810_1.mxl")

QMenOp12 = music21.converter.parse("/content/drive/My Drive/MusicXML/QMenOp12_1.mxl")
QMenOp13 = music21.converter.parse("/content/drive/My Drive/MusicXML/QMenOp13_1.mxl")
QMenOp44No2 = music21.converter.parse("/content/drive/My Drive/MusicXML/QMenOp44No2_1.mxl")
QMenOp80 = music21.converter.parse("/content/drive/My Drive/MusicXML/QMenOp80_1.mxl")
QBocOp39 = music21.converter.parse("/content/drive/My Drive/MusicXML/QBocOp39_1.mxl")
QDvoOp96 = music21.converter.parse("/content/drive/My Drive/MusicXML/QDvoOp96_1.mxl")
QOp59No1 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp59No1_1.mxl")
QOp59No2 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp59No2_1.mxl")
QOp59No3 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp59No3_1.mxl")
QOp18No3 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No3_1.mxl")
QOp18No6 = music21.converter.parse("/content/drive/My Drive/MusicXML/QOp18No6_1.mxl")
QBraOp51No1 = music21.converter.parse("/content/drive/My Drive/MusicXML/QBraOp51No1_1.mxl")
QBraOp51No2 = music21.converter.parse("/content/drive/My Drive/MusicXML/QBraOp51No2_1.mxl")

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
    if len(chunk) == length:
      returnList.append(chunk)
  return returnList

def checkCSV(location):
  df = pd.read_csv(location, header=None)
  print(df.head())

quartets = [QK387, QK421, QK428, QK458, QK464, QK465, QK590, QOp18No4, QOp130, 
            QOp131, QOp18No1, QOp18No2, QOp18No5, QOp74, QOp95, QOp127, QOp132, 
            HobIII33, HobIII34, HobIII63, D667, D810, QMenOp12, QMenOp13, 
            QMenOp44No2, QMenOp80, QBocOp39, QDvoOp96, QOp59No1, QOp59No2, 
            QOp59No3, QOp18No3, QOp18No6, QBraOp51No1, QBraOp51No2]

qPaths = ['QK387', 'QK421', 'QK428', 'QK458', 'QK464', 'QK465', 'QK590', 'QOp18No4', 'QOp130', 'QOp131', 
          'QOp18No1', 'QOp18No2', 'QOp18No5', 'QOp74', 'QOp95', 'Op127', 'QOp132', 'HobIII33', 
          'HobIII34', 'HobIII63', 'D667', 'D810', 'QMenOp12', 'QMenOp13', 'QMenOp44No2', 'QMenOp80', 
          'QBocOp39', 'QDvoOp96', 'QOp59No1', 'QOp59No2', 'QOp59No3', 'QOp18No3', 'QOp18No6', 
          'QBraOp51No1', 'QBraOp51No2']

def compiledList(pieces):
  compiled = []
  for i in pieces:
    compiled.extend(treatScoreQuartet(i))
    print(len(compiled))
  return separateIntoChunks(compiled)

def conTranslateSave(pieces, paths):
  for piece, path in zip(pieces, paths):
    formattedPath = f"/content/drive/My Drive/MusicXML/{path}_R_B.csv"
    translated = treatScoreQuartet(piece)
    separated = separateIntoChunks(translated)
    with open(formattedPath, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(separated)
    checkCSV(formattedPath)

conTranslateSave(quartets, qPaths)
