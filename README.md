# RhythmGen
A rhythmic generation project, using HMM to model the use of rhythm in phrases of various styles, starting with piano sonatas.
The project can be divided roughly into 2 segments, the first is data collection and treatment, and the second is training and use of the model. 
For the data, I'll use free MusicXML files of western classical piano sonatas of the 18th - 19th century, that are currently in the public domain.
I'll use music21 to convert these files and process them to retrieve the melody, after which I'll split them into phrases and convert them into durations
represented as decimals relative to a quarter note. 
Finally I'll write them to csv files.

For the second part, I will use an unsupervised Hidden Markov Model, trained on roughly 1000-2000 phrases with more limited difference in composer and 
style for the proof of concept, and for the proper model, trained on as many as I can find and process, hopefully min 5000. 
I won't be including symbols such as barlines for this, and also won't be restricting it to a single time signiature. I would like the generated phrases to 
sound free and natural, but also not be restricted to the traditional boundaries of the music of the time. I like to think of the generative approach
as somewhat neo-classical in this regard.

I hope to later generate melody over the rhythm, but I'll keep that for later.
