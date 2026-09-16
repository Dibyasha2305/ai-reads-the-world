# AI Reads the World

## Scene Text Recognition in Natural Images

AI Reads the World is a computer vision project that detects and recognizes text appearing in real-world images such as road signs, parking signs, product packaging, and shop boards.

The system uses image preprocessing, EasyOCR-based text detection and recognition, confidence-based filtering, result visualization, and quantitative evaluation to create a modular scene-text recognition pipeline.

---

## Problem Statement

Text appears everywhere in the real world, including road signs, advertisements, product packages, and public information boards. Automatically extracting this text from natural images is challenging because of variations in lighting, scale, orientation, image quality, background complexity, and visual noise.

This project aims to develop a modular computer vision pipeline capable of detecting and recognizing visible text from natural images and evaluating the quality of the OCR results.

---

## Objectives

- Detect text regions from natural scene images.
- Recognize the detected text using an OCR engine.
- Improve input images using preprocessing techniques.
- Filter unreliable OCR detections using confidence scores.
- Visualize detected text using bounding boxes and labels.
- Measure OCR performance using quantitative metrics.
- Maintain a modular and testable software architecture.

---

## Key Features

### 1. Image Loading

The system loads and validates input images using OpenCV.

Supported formats:

- JPG
- JPEG
- PNG
- BMP
- WEBP

### 2. Image Preprocessing

The preprocessing module performs:

- Image resizing
- Noise reduction
- Contrast enhancement using CLAHE

These operations prepare images for more reliable OCR processing.

### 3. Text Detection and Recognition

EasyOCR is used to:

- Detect text regions
- Recognize text
- Generate confidence scores
- Return bounding box coordinates

### 4. Post-Processing

OCR results are processed by:

- Removing empty detections
- Filtering low-confidence results
- Sorting detected text spatially

### 5. Result Visualization

Detected text is displayed directly on the processed image using:

- Bounding polygons
- Recognized text
- Confidence scores

### 6. OCR Evaluation

The project evaluates OCR performance using:

- Character Error Rate (CER)
- Word Accuracy
- Average OCR Confidence
- Number of detections
- Processing time

### 7. Automated Testing

The project includes automated tests using `pytest` for:

- Image loading
- Image resizing
- Denoising
- Contrast enhancement
- Confidence filtering
- CER calculation
- Word accuracy calculation

---

## System Architecture

```text
                    Input Image
                         |
                         v
                  +--------------+
                  | Image Loader |
                  +--------------+
                         |
                         v
                  +--------------+
                  | Preprocessing|
                  +--------------+
                         |
              +----------+----------+
              |          |          |
            Resize    Denoising    CLAHE
              |          |          |
              +----------+----------+
                         |
                         v
                  +--------------+
                  |   EasyOCR    |
                  +--------------+
                         |
                         v
                  +--------------+
                  |Post-Processing|
                  +--------------+
                         |
                         v
                  +--------------+
                  | Visualization|
                  +--------------+
                         |
                         v
                  Annotated Image
                         |
                         v
                  +--------------+
                  |  Evaluation  |
                  +--------------+
                         |
             +-----------+-----------+
             |           |           |
            CER    Word Accuracy  Confidence
Workflow
1. Input image
       ↓
2. Validate and load image
       ↓
3. Resize image if required
       ↓
4. Reduce image noise
       ↓
5. Enhance local contrast
       ↓
6. Detect and recognize text using EasyOCR
       ↓
7. Filter low-confidence detections
       ↓
8. Sort recognized text
       ↓
9. Draw bounding boxes and confidence labels
       ↓
10. Save annotated output
       ↓
11. Evaluate OCR performance
       ↓
12. Save evaluation results as CSV
Technologies Used
Technology	Purpose
Python	Core programming language
OpenCV	Image processing and visualization
EasyOCR	Text detection and recognition
NumPy	Numerical and image array operations
PyTorch	Deep learning framework used by EasyOCR
pytest	Automated testing
CSV	Evaluation result storage
Git/GitHub	Version control and project management
Project Structure
ai-reads-the-world/
│
├── data/
│   └── images/
│       ├── productpackage1.jpg
│       ├── productpackage2.jpg
│       ├── publicsign1.jpg
│       ├── shopsignboard1.jpg
│       ├── streetsign1.jpeg
│       └── streetsign2.jpeg
│
├── results/
│   ├── evaluation_results.csv
│   ├── ground_truth.csv
│   └── annotated outputs
│
├── src/
│   ├── __init__.py
│   ├── image_loader.py
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── postprocessor.py
│   ├── visualizer.py
│   ├── pipeline.py
│   ├── evaluator.py
│   └── experiment.py
│
├── tests/
│   └── test_project.py
│
├── main.py
├── README.md
└── .gitignore
Module Description
image_loader.py

Responsible for loading and validating image files.

preprocessing.py

Contains image preprocessing operations including resizing, denoising, and CLAHE-based contrast enhancement.

ocr_engine.py

Provides an object-oriented interface to the EasyOCR engine.

postprocessor.py

Filters and organizes OCR results according to confidence and spatial ordering.

visualizer.py

Draws detected text regions, recognized text, and confidence scores on images.

pipeline.py

Connects all major components into a single end-to-end scene text recognition pipeline.

evaluator.py

Calculates OCR evaluation metrics including CER, word accuracy, confidence, and processing time.

experiment.py

Runs OCR evaluation over the project image dataset and exports the results to CSV.

Installation
1. Clone the repository
git clone https://github.com/Dibyasha2305/ai-reads-the-world.git
cd ai-reads-the-world
2. Install dependencies
pip install easyocr opencv-python numpy pytest
Running the Project
Run OCR on an image

From the project root:

python main.py

The processed image will be saved inside:

results/
Run the evaluation
python -m src.experiment

Evaluation results will be generated at:

results/evaluation_results.csv
Run automated tests
python -m pytest
Evaluation Metrics
Character Error Rate

Character Error Rate measures the number of character-level edits required to transform the OCR output into the ground-truth text.

CER = Edit Distance / Number of Ground Truth Characters

Lower CER indicates fewer character-level errors.

Word Accuracy

Word-level accuracy evaluates the similarity between the ground-truth word sequence and the OCR-predicted word sequence.

Higher word accuracy indicates better recognition.

Average Confidence

The average confidence score represents the confidence returned by the OCR engine for the detected text regions.

Processing Time

Processing time measures the time required to process an image through the complete OCR pipeline.

Testing

The project contains automated tests using pytest.

Current test coverage includes:

✓ Image loading
✓ Image resizing
✓ Image denoising
✓ Contrast enhancement
✓ Confidence filtering
✓ Character Error Rate
✓ Word Accuracy

Test command:

python -m pytest
Sample Applications

The system can be applied to:

Road sign recognition
Public information extraction
Product packaging analysis
Scene text recognition
Assistive vision systems
Augmented reality applications
Navigation systems
Image-based information extraction
Limitations

The current implementation has several limitations:

OCR performance can decrease on heavily blurred images.
Small or stylized text may be difficult to recognize.
Complex backgrounds can produce false detections.
Watermarks and non-semantic text may also be detected.
The current OCR configuration primarily targets English text.
CPU execution can result in relatively high processing time.
Future Enhancements

Future versions can include:

Multi-language OCR support
GPU acceleration
Perspective correction for angled signs
Advanced text-region filtering
Text orientation detection
Object-aware text detection
Video-based scene text recognition
Real-time camera input
Mobile deployment
Improved ground-truth annotation and benchmarking
Comparison of multiple OCR models
Results

The project evaluates multiple natural scene images containing road signs, public signs, shop boards, and product packaging.

Example evaluation output:

Image                    CER         Word Acc.      Confidence
----------------------------------------------------------------
streetsign2.jpeg         0.00%       100.00%        1.00

Detailed experimental results are stored in:

results/evaluation_results.csv

Note: OCR metrics depend on the ground-truth annotations and image content. Confidence scores should not be interpreted as OCR accuracy.

Conclusion

AI Reads the World demonstrates an end-to-end computer vision pipeline for scene text recognition. The project combines image preprocessing, OCR, post-processing, visualization, quantitative evaluation, and automated testing in a modular architecture.

The modular design makes the system easier to maintain, test, extend, and integrate with future computer vision applications.

Author

Dibyasha