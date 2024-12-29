from numpy import loadtxt
from keras.models import model_from_json     #to save the model
import warnings
warnings.filterwarnings('ignore')

#DATA GATHERING
dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')
print(dataset)


#SPLITTING THE VALUES
x=dataset[:,0:8]   #input
y=dataset[:,8]     #output

json_file=open("model.json",'r')
loadedm=json_file.read()
json_file.close()

model=model_from_json(loadedm)
model.load_weights("models.weights.h5")
print("LOADED MODEL FROM DISK")

#PREDICTING THE MODEL
predictions=model.predict(x)

for i in range(20,25):
    print("%s=>%d(expected %d)"%(x[i].tolist(),predictions[i],y[i]))


                   
