weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy'] # first Feature 
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']# secon features
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']  # Label or target varible
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
# converting string labels into numbers 
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict(features) # 0:Overcast, 2:Mild
print(predicted)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# confusion matrix
matrix = confusion_matrix(predicted,predicted, labels=[0,1])
print('Confusion matrix : \n',matrix)

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(predicted,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(predicted,predicted,labels=[1,0])
print('Classification report : \n',matrix)




