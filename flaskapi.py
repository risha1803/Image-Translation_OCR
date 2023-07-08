# from crypt import methods
from flask import Flask
import codecs
from PIL import Image, ImageDraw, ImageFont  # opening and manipulating images
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras_ocr
import math
from PIL import ImageTk
import requests
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from pytesseract import Output
from flask import jsonify, request , send_file
import uuid 
from flask_restful import Resource, Api
from flask_cors import CORS
 
app= Flask(__name__) 

api = Api(app)
cors = CORS(app)

upload_folder= "C:\\Users\\LENOVO\\Desktop\\New folder1"
app.config['upload_folder']= upload_folder


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded image
    filename= secure_filename(file.filename)
    file.save(os.path.join(app.config['upload_folder'], filename))

    filepath = os.path.join(app.config['upload_folder'], filename)
    print(filepath)
    lang = request.form.get('lang').strip()
    
    inpainttext, translation = text_extraction(filepath, lang)
    print(translation)

    return {'inpaint': inpainttext, 'translation': translation}
    
#, translation.encode('utf-8')
    
# def text_extraction(source_path, intermediate_path, pipeline, source):
def text_extraction(filepath,lang):

    pipeline = keras_ocr.pipeline.Pipeline()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    im = Image.open(filepath)
    im=im.convert("RGB")
    text = pytesseract.image_to_data(im, output_type='data.frame')
    text = text[text.conf != -1]
    df= text[text['conf']>50]
    data= df[['text','left','top','width','height']]
    word=" "
    for value in df['text']:
        word= word + value +" "
    print(word)
    result = translator(word, lang)
    print(result)
    inpainttext =inpaint_text(filepath,pipeline)
    #result = translator(word, lang)
    return inpainttext,result


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(filepath, pipeline):
   
    img = keras_ocr.tools.read(filepath) 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255,    
        thickness)
    inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    image_change= cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    image_pil= Image.fromarray(image_change)
    image_pil.save('C:\\Users\\LENOVO\\Desktop\\New folder\\out.png')
    return 'C:\\Users\\LENOVO\\Desktop\\New folder\\out.png'           

#Translator
def translator(word, lang):
    key = "7f2623d635f944d7969e462d54f8c5cc"
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "centralindia"
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': lang
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

@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.get_json()
    file_path = data['file_path']
    print(file_path)

    if not os.path.isfile(file_path):
        return "File not found", 404

    return send_file(file_path, mimetype='image/jpeg')

@app.route('/final-image', methods=['POST'])
def final_image():
    if 'file' not in request.files:
        return "No file uploaded", 400    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded image
    filename= secure_filename(file.filename)
    file.save(os.path.join(app.config['upload_folder'], filename))

    filepath = os.path.join(app.config['upload_folder'], filename)
    
    lang = request.form.get('lang').strip()

    image= image_translation(filepath, lang)
    return send_file(image, mimetype='image/jpeg')

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

def put_text(path, final_list, source):
    img = Image.open(path)
    I1 = ImageDraw.Draw(img)
    font_files = {
    'hi': 'NirmalaB.ttf'
    
    #,'en': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Arial.ttf',
    #'bn': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\bengali.ttf',
    #'ar-sa': 'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Arabic.ttf',
    #'fr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\french.ttf',
    #'gu':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Gujrati.ttf',
    #'de':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\German.ttf',
    #'ja':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Japanese.ttf',
    #'kn':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Kannada.ttf',
    #'ko':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Korean.ttf',
    #'ml':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Malayalam.ttf',
    #'mr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Marathi.ttf',
    #'ta':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Tamil.ttf',
    #'te':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\NotoSansTelugu-VariableFont_wdth,wght.ttf',
    #'th':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Thai.ttf',
    #'tr':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Turkish.ttf',
    #'vi':'C:\\Users\\LENOVO\\Desktop\\prisha.aicte\\Vitnamese.ttf'
    
    }
    # Add more languages and their corresponding font files here
    

    ttf = font_files[source]
    print(final_list)
    draw = ImageDraw.Draw(img)
    for height, top, left, word in final_list:
        myFont = ImageFont.truetype(ttf, height + 2, layout_engine=ImageFont.Layout.RAQM)
        text = str(word)
        coordinate= (top, left)
        draw.text(coordinate, text, fill=(0, 0, 0), font=myFont)
    img.save('C:\\Users\\LENOVO\\Desktop\\New folder\\out1.png')

    return 'C:\\Users\\LENOVO\\Desktop\\New folder\\out1.png'

def image_translation(filepath, lang):
    pipeline = keras_ocr.pipeline.Pipeline()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    im = Image.open(filepath)
    im=im.convert("RGB")
    #im.show()
    width, height = im.size
    print(width)
    text = pytesseract.image_to_data(im, output_type='data.frame')
    text = text[text.conf != -1]
    df= text[text['conf']>50]
    data= df[['text','left','top','width','height']]
    
    result_list = []
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

    final_list2=[]
    for sublist in result_list:
        height, left, top, sentences = sublist
        word = translator(sentences, 'hi')
        final_list2.append([height, left, top, word])   

    final_list=[]

    for item in final_list2:
        parts= break_sentence(item[3], width*1.75, item[0])
        current_y= item[2]
        for part in parts:
            final_list.append([item[0], item[1], current_y, part])
            current_y= current_y + item[0] + 10

    img = inpaint_text(filepath, keras_ocr.pipeline.Pipeline())
    image= put_text(img, final_list, lang)
    return final_list



if __name__ == "__main__":
	#app.run(debug = True, host='0.0.0.0', port =3000)
    app.run(debug = True,use_reloader=False)