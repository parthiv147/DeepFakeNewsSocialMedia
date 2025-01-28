from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Convolution2D
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
import io
import base64

global accuracy, precision, recall, fscore, X, Y, sc, tfidf_vectorizer
accuracy = []
precision = []
recall = []
fscore = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

if os.path.exists("model/X.npy"):
    f = open('model/tfidf.pckl', 'rb')
    tfidf_vectorizer = pickle.load(f)
    f.close()
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
else:
    dataset = pd.read_csv("Dataset/Tweepfake.csv", sep=";")
    dataset = dataset.dropna()
    dataset = dataset.values
    X = []
    Y = []
    for i in range(len(dataset)):
        tweet = dataset[i, 1]
        tweet = tweet.strip("\n").strip().lower()
        label = dataset[i, 2]
        tweet = cleanText(tweet)
        X.append(tweet)
        Y.append(1 if label == 'bot' else 0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=900)
    X = tfidf_vectorizer.fit_transform(X).toarray()
    with open('model/tfidf.pckl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    np.save("model/X", X)
    np.save("model/Y", Y)

sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

nb_cls = GaussianNB()
nb_cls.fit(X_train[:1000], y_train[:1000])
predict = nb_cls.predict(X_test[:200])
calculateMetrics("Naive Bayes", predict, y_test[:200])

lr_cls = LogisticRegression(max_iter=300)
lr_cls.fit(X_train[:1000], y_train[:1000])
predict = lr_cls.predict(X_test[:200])
calculateMetrics("Logistic Regression", predict, y_test[:200])

dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train[:1000], y_train[:1000])
predict = dt_cls.predict(X_test[:200])
calculateMetrics("Decision Tree", predict, y_test[:200])

rf_cls = RandomForestClassifier()
rf_cls.fit(X_train[:1000], y_train[:1000])
predict = rf_cls.predict(X_test[:200])
calculateMetrics("Random Forest", predict, y_test[:200])

gb_cls = GradientBoostingClassifier()
gb_cls.fit(X_train[:1000], y_train[:1000])
predict = gb_cls.predict(X_test[:200])
calculateMetrics("Gradient Boosting", predict, y_test[:200])


X_train1 = np.reshape(X_train, (X_train.shape[0], 30, 10, 3))
X_test1 = np.reshape(X_test, (X_test.shape[0], 30, 10, 3))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 50, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
calculateMetrics("CNN Algorithm", predict, y_test1)

hybrid_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)
hybrid_features = hybrid_model.predict(X_test1)
print(hybrid_features.shape)
Y = y_test
X_train, X_test, y_train, y_test = train_test_split(hybrid_features, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(hybrid_features, Y, test_size=0.1)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("Extension Hybrid CNN", predict, y_test)

def LoadDataset(request):
    if request.method == 'GET':
        dataset = pd.read_csv("Dataset/Tweepfake.csv", sep=";")
        dataset = dataset.dropna()
        dataset = dataset.values
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Tweet</th><th><font size="" color="black">Account Type</th>'
        output+='</tr>'
        for i in range(0, 100):
            output += '<tr><td><font size="" color="black">'+dataset[i,0]+"</td>"
            output += '<td><font size="" color="black">'+dataset[i,1]+"</td>"
            output += '<td><font size="" color="black">'+dataset[i,2]+"</td></tr>"
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)      

def FastText(request):
    if request.method == 'GET':
        global X
        context= {'data':str(X)}
        return render(request, 'ViewOutput.html', context)

def TrainML(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        algorithms = ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Propose CNN', 'Extension Hybrid CNN']
        output = "<table border=1 align=center width=100%><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F-Score</th></tr>"
        for i in range(len(algorithms)):
            output += f'<tr><td>{algorithms[i]}</td><td>{accuracy[i]}</td><td>{precision[i]}</td><td>{recall[i]}</td><td>{fscore[i]}</td></tr>'
        output += '</table><br/>'

        fig, ax = plt.subplots(figsize=(10, 6))
        index = np.arange(len(algorithms))
        bar_width = 0.2
        ax.bar(index, accuracy, bar_width, label='Accuracy', color='b')
        ax.bar(index + bar_width, precision, bar_width, label='Precision', color='g')
        ax.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='r')
        ax.bar(index + 3 * bar_width, fscore, bar_width, label='F-Score', color='y')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Metrics')
        ax.set_title('Algorithms Performance - Grouped Bar Graph')
        ax.set_xticks(index + 1.5 * bar_width)
        ax.set_xticklabels(algorithms)
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        grouped_bar_img = base64.b64encode(buf.getvalue()).decode()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(algorithms, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')
        ax.plot(algorithms, precision, marker='o', linestyle='-', color='g', label='Precision')
        ax.plot(algorithms, recall, marker='o', linestyle='-', color='r', label='Recall')
        ax.plot(algorithms, fscore, marker='o', linestyle='-', color='y', label='F-Score')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Metrics')
        ax.set_title('Algorithms Performance - Line Graph')
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        line_img = base64.b64encode(buf.getvalue()).decode()

        context = {
            'data': output,
            'grouped_bar_img': grouped_bar_img,
            'line_img': line_img
        }
        return render(request, 'ViewGraph.html', context)

def DetectFakeAction(request):
    if request.method == 'POST':
        global username, hybrid_model, tfidf_vectorizer, sc
        tweet = request.POST.get('t1', False)
        data = tweet.strip().lower()
        data = cleanText(data)
        temp = []
        temp.append(data)
        temp = tfidf_vectorizer.transform(temp).toarray()
        dl_model = load_model("model/cnn_weights.hdf5")
        temp = sc.transform(temp)
        temp = np.reshape(temp, (temp.shape[0], 30, 10, 3))
        predict = dl_model.predict(temp)
        predict = np.argmax(predict)
        print(predict)
        output = "Normal"
        if predict == 1:
            output = "Bot Fake"
        context= {'data': 'Given Tweet Detected as : '+output}
        return render(request, 'DetectFake.html', context)         

def DetectFake(request):
    if request.method == 'GET':
       return render(request, 'DetectFake.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "UserLogin.html"
        context= {'data':'Invalid login details'}                      
        if "deepfake.com" == username and "human" == password:
            context = {'data':"Welcome "+username}
            status = 'UserScreen.html'            
        return render(request, status, context)              


    
