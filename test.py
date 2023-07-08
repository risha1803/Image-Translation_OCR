import pytesseract
from PIL import Image, ImageDraw, ImageFont  # opening and manipulating images
from pytesseract import Output
import fileinput
import argparse
import numpy as np
import pandas as pd
import keras_ocr
import json
import cv2
import uuid
import requests
#import requests
#import uuid
#import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
im = Image.open("C:\\Users\\LENOVO\\Desktop\\New folder1\\photo5.jpg")
im=im.convert("RGB")
width, height = im.size

# Print the dimensions
extracted_text = pytesseract.image_to_string(im, lang='eng', config='--psm 3')
data = pytesseract.image_to_data(im, output_type=Output.DICT, lang='eng', config='--psm 3')
data1 = pytesseract.image_to_data(im, lang='eng', config='--psm 3')
#print(extracted_text)


locations={}
text = pytesseract.image_to_data(im, output_type='data.frame')
text = text[text.conf != -1]


df= text[text['conf']>50]
data= df[['text','left','top','width','height']]


# Initialize the list to store the results
result_list = []

# Iterate over unique combinations of block_num and par_num
for _, group in df.groupby(["block_num", "par_num"]):
    # Extract the block_num, par_num, height, left, and top values from the first word
    block_num = group["block_num"].iloc[0]
    par_num = group["par_num"].iloc[0]
    height = group["height"].iloc[0]
    left = group["left"].iloc[0]
    top = group["top"].iloc[0]
    # Join all the words in the block and para number to form sentences
    sentences = " ".join(group["text"])
    # Append the information to the result list
    result_list.append([height, left, top, sentences])

# Print the result list
for item in result_list:
    print(item)

def translator(word, source):
    key = "7f2623d635f944d7969e462d54f8c5cc"
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "centralindia"
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': source
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': word}]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    translated_word = response[0]["translations"][0]["text"]
    return translated_word

final_list2=[]

for sublist in result_list:
    
    for sublist in result_list:
        height, left, top, sentences = sublist
        word = translator(sentences, 'hi')
        final_list2.append([height, left, top, word])

for item in final_list2:
    print(item)

def break_sentence(sentence, max_width, font_size):
    words = sentence.split()
    parts = []
    current_part = ""
    for word in words:
        if current_part:
            # Check if adding the next word exceeds the maximum width
            if (len(current_part) + len(word))*font_size*1.2 + 1 > max_width:
                parts.append(current_part)
                current_part = word
            else:
                current_part += " " + word
        else:
            current_part = word

    if current_part:
        parts.append(current_part)

    return parts

final_list=[]

for item in final_list2:
    parts= break_sentence(item[3], width*1.75, item[0])
    current_y= item[2]
    for part in parts:
        final_list.append([item[0], item[1], current_y, part])
        current_y= current_y + item[0] + 10
# Print the DataFrame
#print(df)

def put_text(path, final_list, source):
    img = Image.open(path)
    I1 = ImageDraw.Draw(img)
    font_files = {
    'hi': 'NirmalaB.ttf',
    'en': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Arial.ttf',
    'bn': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\bengali.ttf',
    'ar-sa': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Arabic.ttf',
    'fr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\french.ttf',
    'gu':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Gujrati.ttf',
    'de':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\German.ttf',
    'ja':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Japanese.ttf',
    'kn':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Kannada.ttf',
    'ko':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Korean.ttf',
    'ml':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Malayalam.ttf',
    'mr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Marathi.ttf',
    'ta':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Tamil.ttf',
    'te':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\NotoSansTelugu-VariableFont_wdth,wght.ttf',
    'th':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Thai.ttf',
    'tr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Turkish.ttf',
    'vi':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Vitnamese.ttf'
    }
    # Add more languages and their corresponding font files here
    

    ttf = font_files[source]
    print(ttf)
    draw = ImageDraw.Draw(img)

    for height, top, left, word in final_list:
        myFont = ImageFont.truetype(ttf, height + 2, layout_engine=ImageFont.Layout.RAQM)
        text = str(word)
        coordinate= (top, left)
        
        draw.text(coordinate, text, fill=(0, 0, 0), font=myFont)

        

    return img


import matplotlib.pyplot as plt

import cv2
import math
import numpy as np
import pytesseract


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path):
    # read the image 
    img = keras_ocr.tools.read(img_path)
    pipeline = keras_ocr.pipeline.Pipeline() 
   
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    blurred_img = cv2.medianBlur(inpainted_img, 9)            

    cv2.imwrite("C:\\Users\\LENOVO\\Desktop\\New folder1\\output_image.jpg", blurred_img)
    return(blurred_img)
    

img = inpaint_text("C:\\Users\\LENOVO\\Desktop\\New folder1\\photo5.jpg")
image= put_text("C:\\Users\\LENOVO\\Desktop\\New folder1\\output_image.jpg", final_list, 'hi')

window_name='image' 
# Using cv2.imshow() method
# Displaying the image
image = np.array(image)
cv2.imshow(window_name, image)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()




