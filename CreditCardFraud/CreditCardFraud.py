
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

main = tkinter.Tk()
main.title("ALGORITHM FOR CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING TECHNIQUES")
main.geometry("1300x1200")

global filename
global accuracy
global X, Y
global X_train, X_test, y_train, y_test
global kmeans

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def processDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset = dataset.sample(frac=1)#randomize the whole dataset
    X = dataset.drop(["Time","Class"],axis=1)
    Y = pd.DataFrame(dataset[["Class"]])
    X = X.values
    Y = Y.values
    X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.90)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    text.insert(END,'Dataset contains total records : '+str(len(X))+"\n")
    text.insert(END,"Application using 80% dataset records to train KMEANS : "+str(len(X_train))+"\n")
    text.insert(END,"Application using 20% dataset records to test KMEANS  : "+str(len(X_test))+"\n")    


def runKMEANS():
    global accuracy
    global kmeans
    kmeans = KMeans(n_clusters=2,random_state=0,algorithm="elkan",max_iter=10000,n_jobs=-1)
    kmeans.fit(X_train)
    predict = kmeans.predict(X_test)
    accuracy = accuracy_score(y_test,predict)
    text.insert(END,'\nkMEANS Prediction Accuracy : '+str(accuracy)+"\n")

    plt.scatter(X_test[:, 0], X_test[:, 1], c=predict, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

def evaluateTransaction():
    text.delete('1.0', END)
    test = filedialog.askopenfilename(initialdir="dataset")
    dataset = pd.read_csv(test)
    dataset = dataset.drop(["Time"],axis=1)
    dataset = dataset.values
    dataset = normalize(dataset)
    predict = kmeans.predict(dataset)                                 
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"X=%s, Predicted = %s" % (dataset[i], 'Transaction Contains Cleaned Signatures')+"\n\n")
        if predict[i] == 1:
            text.insert(END,"X=%s, Predicted = %s" % (dataset[i], 'Transaction Contains Fraud Transaction Signature')+"\n\n")
     
def graph():
    global accuracy
    accuracy = accuracy * 100
    error = 100 - (accuracy)
    height = [accuracy,error]
    bars = ('KMeans Correct Prediction Accuracy','Incorrect Prediction Error Rate')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def close():
    main.destroy()
    
font = ('times', 14, 'bold')
title = Label(main, text='ALGORITHM FOR CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING TECHNIQUES')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=310,y=100)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

kmeansButton = Button(main, text="Apply KMeans Algorithm", command=runKMEANS)
kmeansButton.place(x=310,y=150)
kmeansButton.config(font=font1) 

evaluate = Button(main, text="Upload Test Transaction & Evaluate Risk Zone", command=evaluateTransaction)
evaluate.place(x=580,y=150)
evaluate.config(font=font1) 

graphbutton = Button(main, text="Accuracy & Error Rate Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

exitb = Button(main, text="Exit", command=close)
exitb.place(x=310,y=200)
exitb.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
