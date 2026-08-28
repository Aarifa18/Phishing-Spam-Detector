#data libraries
import pandas as pd
import os
import joblib
#model libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
#Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#drawing confusion matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from text_preprocessing import text_processing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data","spam_email.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
data = pd.read_csv(DATA_PATH)


#Splitting data in to training and testing data
#70% - training, 30% - testing
data['input_text'] = data['subject'].fillna(' ') + ' ' + data['body']
X = data['input_text']
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

model = svm.SVC(kernel='poly', degree=8) #polynomial kernel SVM - higher accuracy
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=text_processing, token_pattern=None)),
    ('tfidf', TfidfTransformer()),
    ('classifier', model)
])

#training pipeline
pipeline.fit(X_train, y_train)
#testing the pipeline
pred = pipeline.predict(X_test)

joblib.dump(pipeline, os.path.join(MODEL_DIR,"nlp_pipeline.joblib"))
print(f"Model saved to {os.path.join(MODEL_DIR,'nlp_pipeline.joblib')}")

#evaluation
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", round(accuracy,2))
print("Other Metrics:")
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

 #displaying confusion matrix
conf_matrix = confusion_matrix(y_test, pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()