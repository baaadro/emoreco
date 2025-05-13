# emoreco
This project aims to create a display capable of automatically detecting the face of the user aswell as guessing the emotion on their face.


This project trains a model by itself using FER-2013 "https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition?resource=download" (dependency)


the training approximately takes 30-40 minutes by default as it runs the training on 20 epochs but the amount can be further increased for better accuracy. (line 59)


after the training the model is automatically saved and it can be reused using the python code in the pretrained folder
