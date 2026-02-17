AI That Reads the World  

Scene Text Recognition in Natural Images



Overview

This project builds a scene text recognition system capable of detecting and reading text from real-world images such as product packaging, street signs, and public signage.



The system uses deep learning–based OCR to locate text regions and recognize words in natural environments with varying lighting, fonts, and backgrounds.







Features

\- Detects text regions in natural scene images  

\- Recognizes words using OCR  

\- Draws bounding boxes and labels  

\- Processes multiple images automatically  

\- Saves annotated outputs  







Example Results



Product Packaging

!\[Packaging](results/productpackage1\_detected.jpg)



Public Sign

!\[Public Sign](results/publicsign1\_detected.jpg)



Street Sign

!\[Street Sign](results/streetsign1\_detected.jpg)







Method

The pipeline consists of:



1\. Load scene image  

2\. Detect text regions using EasyOCR  

3\. Recognize text content  

4\. Draw bounding boxes and labels  

5\. Save annotated output  





Tech Stack

\- Python  

\- EasyOCR  

\- OpenCV  

\- NumPy  

\- Matplotlib  







Project Structure

ai-reads-the-world/

│

├── data/images/

├── results/

├── src/detect\_text\_easyocr.py

└── README.md







Applications

\- Scene OCR  

\- Assistive vision systems  

\- Autonomous navigation  

\- Augmented reality text reading  

\- Document AI  







Author

Dibyasha



