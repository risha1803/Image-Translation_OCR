# Image-Translation_OCR
This code provided is can be used for translating an image from one language to another.

While working with an image or a file containing multiple images as input, I utilized OCR models such as pytesseract and keras_ocr for extracting text from the images. These OCR models employ advanced techniques to recognize and extract text from images.

Once the text is extracted, I then proceed with translating the text using translation libraries or APIs. Translation services like Google Translate or libraries such as translate-python can be used to achieve this.

To maintain the original structure of the image and preserve other elements apart from the text, I generate new images with the translated text overlaid on them. This can be done by leveraging image processing libraries like OpenCV or PIL (Python Imaging Library).

By combining these steps, we were successful in creating a pipeline that takes an image or a file with images as input, extracts the text using OCR models, performs translation on the extracted text, and generates new images with the translated text while preserving the original structure and other elements of the image.

