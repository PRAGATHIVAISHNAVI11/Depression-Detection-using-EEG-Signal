from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Flatten, Convolution2D
from keras.models import Sequential
import pickle
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Depression Detection & Survey-Based Recommendation")
main.geometry("1000x700")

global dataset, X, Y, cnn, selected_record

def loadDataset():
    global dataset
    filename = filedialog.askopenfilename()
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.Label[dataset.Label == 2.0] = 1.0
    featuresExtraction()

def featuresExtraction():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Extracted Features from EEG Signals\n\n")
    text.insert(END,"Total features found in each record: "+str(X.shape[1])+"\n")
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records used to train: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test: "+str(X_test.shape[0])+"\n")
    runCNN()

def runCNN():
    global cnn
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        cnn.load_weights("model/model_weights.h5")
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(18, 18, 3)))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(256, activation='relu'))
        cnn.add(Dense(2, activation='softmax'))
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    text.insert(END, "CNN Model Run Successfully!\n")

def predictDepression():
    global cnn, selected_record
    labels = ["Normal", "Depressed"]
    filename = filedialog.askopenfilename()
    text.delete('1.0', END)
    text.insert(END, filename + " loaded for testing\n\n")
    dataset = pd.read_csv(filename)
    dataset = dataset.values
    testData = dataset[:, 0:972]
    test_X = testData.reshape(testData.shape[0], 18, 18, 3)
    predict = cnn.predict(test_X)
    predict = np.argmax(predict, axis=1)
    
    results = []
    for i in range(len(testData)):
        result = f"EEG Record {i + 1}: PREDICTED AS ===> {labels[predict[i]]}\n"
        results.append(result)
    text.insert(END, "\n".join(results))
    
    selected_record.set("Select EEG Record")
    record_menu = OptionMenu(main, selected_record, *[f"EEG Record {i+1}" for i in range(len(testData))])
    record_menu.place(x=400, y=50)
    surveyButton.place(x=600, y=50)

def survey():
    record = selected_record.get()
    if record == "Select EEG Record":
        text.insert(END, "Please select a record for the survey!\n")
        return
    record_id = int(record.split()[-1])
    survey_window(record_id)

def survey_window(record_id):
    survey_frame = Toplevel(main)
    survey_frame.title(f"Survey for EEG Record {record_id}")
    survey_frame.geometry("400x300")
    
    Label(survey_frame, text=f"EEG Record {record_id}: Survey", font=("Arial", 12, "bold")).pack()
    responses = []
    questions = [
        "Have you been feeling down or depressed? (1-5)",
        "Do you have trouble sleeping? (1-5)",
        "Do you feel tired or have low energy? (1-5)",
        "Have you lost interest in activities? (1-5)",
    ]
    for q in questions:
        frame = Frame(survey_frame)
        frame.pack()
        Label(frame, text=q).pack(side=LEFT)
        var = IntVar(value=3)
        responses.append(var)
        Scale(frame, from_=1, to=5, orient=HORIZONTAL, variable=var).pack(side=RIGHT)
    
    Button(survey_frame, text="Submit", command=lambda: submitSurvey(record_id, responses, survey_frame)).pack()

def submitSurvey(record_id, responses, window):
    scores = [var.get() for var in responses]
    avg_score = np.mean(scores)
    text.insert(END, f"Survey for EEG Record {record_id} Completed. Average Score: {avg_score}\n")
    recommend(avg_score)
    window.destroy()

def recommend(score):
    if score < 2:
        rec = "You're doing well! Keep up with healthy habits."
    elif score < 4: 
        rec = "Consider mindfulness, exercise, and social activities."
    else:
        rec = "You might benefit from therapy or guided mental health activities."
    text.insert(END, f"Recommendation: {rec}\n")

selected_record = StringVar()
surveyButton = Button(main, text="Take Survey", command=survey)

title = Label(main, text='Depression Detection using EEG', font=('times', 16, 'bold'), bg='lavender blush', fg='DarkOrchid1')
title.pack()

uploadButton = Button(main, text="Upload EEG Data", command=loadDataset)
uploadButton.place(x=10, y=50)

predictButton = Button(main, text="Predict Depression", command=predictDepression)
predictButton.place(x=200, y=50)

text = Text(main, height=20, width=120)
text.place(x=10, y=100)

main.mainloop()
