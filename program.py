 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  #ignoring the warnings

#reading the dataset
df=pd.read_csv("hearts.csv")
print(df)

#we should convert the letters to numeric so we use labelencoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#converting using fittransformer function
df['Gender']=le.fit_transform(df['Gender'])
df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
df.to_csv('transformed_data.csv', index=False)
print(df)

#data splitting
#defining input and output
x=df.drop(columns=['HeartDisease'])  #input
y=df['HeartDisease']                 #output
print(x)
print(y)

#SPLIT THE DATA INTO TRAIN AND TEST SETS
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)

#MODEL TRAINING
#IMPORT CLASSIFIERS

from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)        #MODEL EVALUATION
print(y_pred)
print(y_test)

#NOW WE SHOULD COMPARE TO FIND ACCURACY
from sklearn.metrics import accuracy_score
print("ACCURACY IS....:",accuracy_score(y_test,y_pred))

#SAVING THE MODEL
import pickle
pickle.dump(NB,open("model.pkl","wb"))

            
#MODEL PREDICTION
#TEST PREDICTION WITH NEW INPUT USING NAIVE BAYES
testprediction=NB.predict([[50,1,0,145,0,1,1,139,1,0.7,1]])
if testprediction==1:
    print("THE PATIENT HAS HEART DISEASE!, PLEASE CONSULT THE DOCTOR!!")
else:
    print("THE PATIENT IS NORMAL")
    

 


