import tkinter as tk
from tkinter import filedialog
from tkinter import *

from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
import cv2


def facial_expression_model(json_file,weights_file):
    with open(json_file,'r') as file:
        loaded_model_json=file.read()
        model=model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

top=tk.Tk()
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1=Label(top,background='#CDCDCD',font=('arial',15,'bold'))
sign_image=Label(top)

facec=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=facial_expression_model('model_a1.json','weights.weights1.h5')

emotion_list=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def detect(file_path):
    global Label_packed

    image=cv2.imread(file_path)
    grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=facec.detectMultiScale(grey_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc=grey_image[y:y+h,x:x+w]
            roi=cv2.resize(fc,(48,48))
            prediction=emotion_list[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print('Predicted emotion is:',prediction)
            label1.configure(foreground='#011638',text=prediction)
    except:
        label1.configure(foreground='#011638',text='Unable to Predict')


def show_detect_button(file_path):
    detect_b=Button(top,text='detect emotion',command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfile()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3),(top.winfo_height()/2.3)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        show_detect_button(file_path)

    except:
        pass

upload=Button(top,text='upload image',command=upload_image,padx=10,pady=5)
upload.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='bottom',expand='True')
heading=Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()