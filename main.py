# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:00:50 2021

@author: tahsi
"""

import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

#header =st.beta_container()


#ith header:
st.title('Interpretation of American Sign Language Using CNN')
st.text('For deaf and mute people, we can use computer vision to correctly generate\nalphabets based on sign language symbols produced by hands.')    
    

st.image(Image.open('ASL.jpg'),caption='ASL images examples')    
##Function for applying Transformation to images
loader =transforms.Compose(
    [ 
        transforms.Resize((224,224)),
        transforms.ToTensor(), #Converting to tensors
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])#Normalizing 
    ])

def image_loader(image_name):  
    image = loader(image_name).float()
    image = image.unsqueeze(0)  
    return image 

@st.cache() #This is to load the function in streamlitt
def load_model():
  model = models.resnet18(pretrained=True)
  model.fc = nn.Linear(in_features = 512, out_features = 29)
  device = torch.device('cpu')
  model.load_state_dict(torch.load('checkpoint.pth',map_location=device)) 
  model.eval()
  return model

model = load_model()
  
label = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
      
#image = image_loader('idk.jp   g')

file = st.file_uploader("Now go ahead upload your own picture in jpg format", type='jpg')
if file is None:
    st.text("please upload an image")
else:
    image = Image.open(file)
    st.image(image,"Uploaded Sign Language Image",use_column_width=True)
    prediction = image_loader(image)
    score1 = model(prediction)
    score1.argmax()
    string = "This is most likely:" +label[score1.argmax()]
    st.success(string)
    
