from numpy import loadtxt                    #to read the data
from keras.models import Sequential           #to arrange the dats in order
from keras.layers import Dense               #to connect the layers
from keras.models import model_from_json     #to save the model

#DATA GATHERING
dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')
print(dataset)


#SPLITTING THE VALUES
x=dataset[:,0:8]   #input
y=dataset[:,8]     #output

print("INPUT:",x)
print("OUTPUT:",y)

#BUILDING THE NEURAL NETWORK
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#MODEL TRAINING
model.fit(x,y,epochs=10,batch_size=5)

#MODEL EVALUATION
_,accuracy=model.evaluate(x,y)
print("ACCURACY: %2f"%(accuracy*100))

#MODEL SAVIING
model_json=model.to_json()
with open("model.json","w")as json_file:
    json_file.write(model_json)

 #saving the weights of the node into a file   
model.save_weights("models.weights.h5")
print("Saved model to disk")
                   







