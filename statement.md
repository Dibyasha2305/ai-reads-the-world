# Project Statement

## Project Title

**AI Reads the World – Scene Text Recognition in Natural Images**

---

## 1. Problem Statement

Text is an important source of information in real-world environments. It can appear on road signs, public information boards, product packaging, shop boards, and other objects.

Manually reading and extracting such information from images is not always practical. Computer vision and Optical Character Recognition (OCR) can be used to automatically detect and recognize text from natural scene images.

However, scene text recognition is challenging because real-world images may contain complex backgrounds, varying lighting conditions, different text sizes, noise, blur, and non-standard orientations.

This project develops a modular computer vision system that processes natural images, detects text regions, recognizes their content, filters unreliable detections, visualizes the results, and evaluates OCR performance.

---

## 2. Aim

To develop an end-to-end computer vision pipeline for detecting and recognizing text from natural scene images using image preprocessing and Optical Character Recognition.

---

## 3. Objectives

1. To load and validate natural scene images.
2. To preprocess images using resizing, denoising, and contrast enhancement.
3. To detect text regions using EasyOCR.
4. To recognize text appearing in natural images.
5. To filter low-confidence OCR detections.
6. To visualize recognized text using bounding boxes and confidence scores.
7. To evaluate OCR performance using Character Error Rate and Word Accuracy.
8. To measure processing time and OCR confidence.
9. To develop automated tests for important system components.
10. To maintain the project using modular software architecture and Git.

---

## 4. Functional Requirements

### FR1 – Image Input

The system shall accept supported image formats including JPG, JPEG, PNG, BMP, and WEBP.

### FR2 – Image Validation

The system shall verify that the input image exists, uses a supported format, and can be successfully loaded.

### FR3 – Image Preprocessing

The system shall resize images when necessary and apply denoising and contrast enhancement.

### FR4 – Text Detection and Recognition

The system shall detect text regions and recognize text using an OCR engine.

### FR5 – Confidence Filtering

The system shall remove OCR detections below a configurable confidence threshold.

### FR6 – Result Visualization

The system shall display recognized text, confidence values, and bounding regions on the processed image.

### FR7 – Evaluation

The system shall calculate OCR evaluation metrics including Character Error Rate, Word Accuracy, average confidence, detection count, and processing time.

### FR8 – Result Storage

The system shall save evaluation results in CSV format.

---

## 5. Non-Functional Requirements

### NFR1 – Performance

The system should process images within a reasonable amount of time on a standard CPU-based system.

### NFR2 – Reliability

The system should handle missing, unsupported, or unreadable image files without silently failing.

### NFR3 – Maintainability

The system should use separate modules for image loading, preprocessing, OCR, post-processing, visualization, evaluation, and experimentation.

### NFR4 – Testability

Core functionality should be independently testable using automated unit tests.

### NFR5 – Usability

The system should provide clear terminal output and save processed results in an easily accessible format.

### NFR6 – Extensibility

The architecture should allow additional OCR engines, preprocessing techniques, languages, and evaluation metrics to be incorporated in future versions.

---

## 6. Major Functional Modules

The system is divided into the following major modules:

### 1. Image Loader

Responsible for loading and validating input images.

**File:** `src/image_loader.py`

### 2. Preprocessing Module

Responsible for image resizing, denoising, and contrast enhancement.

**File:** `src/preprocessing.py`

### 3. OCR Engine

Provides the interface to EasyOCR for text detection and recognition.

**File:** `src/ocr_engine.py`

### 4. Post-Processing Module

Filters low-confidence results and organizes recognized text.

**File:** `src/postprocessor.py`

### 5. Visualization Module

Draws text regions and recognition results on the image.

**File:** `src/visualizer.py`

### 6. Pipeline Module

Integrates all major processing stages into a complete workflow.

**File:** `src/pipeline.py`

### 7. Evaluation Module

Calculates OCR performance metrics.

**File:** `src/evaluator.py`

### 8. Experiment Module

Runs evaluation across the project image dataset and stores results.

**File:** `src/experiment.py`

---

## 7. Input

The system accepts natural scene images such as:

- Road signs
- Parking signs
- Product packages
- Shop boards
- Public information signs

Example:

```text
data/images/streetsign1.jpeg
8. Output

The system produces:

Annotated Images

Images containing:

Detected text boundaries
Recognized text
OCR confidence scores
Evaluation Results

A CSV file containing:

Image name
Ground-truth text
Predicted text
Number of detections
Character Error Rate
Word Accuracy
Average confidence
Processing time

Output:

results/evaluation_results.csv
9. System Workflow
Input Image
     |
     v
Image Validation
     |
     v
Image Loading
     |
     v
Preprocessing
     |
     +----> Resize
     |
     +----> Denoising
     |
     +----> CLAHE Contrast Enhancement
     |
     v
EasyOCR
     |
     v
Text Detection + Recognition
     |
     v
Confidence Filtering
     |
     v
Text Sorting
     |
     v
Visualization
     |
     v
Annotated Image
     |
     v
Evaluation
     |
     +----> CER
     |
     +----> Word Accuracy
     |
     +----> Confidence
     |
     +----> Processing Time
     |
     v
CSV Results
10. Technical Concepts Used

The project applies the following concepts:

Computer Vision
Image preprocessing
Image resizing
Image denoising
CLAHE
Optical Character Recognition
Text detection
Text recognition
Bounding boxes
Confidence thresholding
Levenshtein distance
Character Error Rate
Word-level evaluation
Object-oriented programming
Modular software architecture
Unit testing
CSV-based data storage
Git version control
11. Software Architecture

The project follows a modular pipeline architecture.

                 +----------------+
                 |    main.py     |
                 +-------+--------+
                         |
                         v
                 +----------------+
                 |    Pipeline    |
                 +-------+--------+
                         |
       +-----------------+-----------------+
       |                 |                 |
       v                 v                 v
 Image Loader      Preprocessing       OCR Engine
       |                 |                 |
       +-----------------+-----------------+
                         |
                         v
                 Post-Processing
                         |
                         v
                    Visualizer
                         |
                         v
                    Evaluation
                         |
                         v
                  CSV Results
12. Testing Strategy

Automated unit tests are implemented using pytest.

The test suite verifies:

Image loading
Image resizing
Image denoising
Contrast enhancement
Confidence filtering
Character Error Rate
Word Accuracy

The current test suite contains 7 automated tests.

Test command:

python -m pytest
13. Expected Benefits

The system provides an automated approach for extracting textual information from natural scene images.

Potential applications include:

Assistive vision
Road sign understanding
Navigation systems
Product information extraction
Public information extraction
Augmented reality
Scene understanding
14. Limitations

The current system may experience reduced performance when:

Text is heavily blurred.
Text is extremely small.
Images contain complex backgrounds.
Text is partially occluded.
Lighting conditions are poor.
Text is highly stylized.
Watermarks or unrelated text appear in the image.

The current configuration primarily uses English OCR.

15. Future Scope

Future development may include:

Multi-language OCR.
Real-time camera-based recognition.
Video scene-text recognition.
GPU acceleration.
Perspective correction.
Text orientation detection.
Advanced false-positive filtering.
Comparison of multiple OCR models.
Mobile deployment.
Improved benchmark datasets and annotation.
16. Conclusion

AI Reads the World provides a complete modular pipeline for scene text recognition from natural images.

The project combines computer vision preprocessing, OCR, post-processing, visualization, quantitative evaluation, and automated testing.

Its modular architecture provides a foundation for extending the system toward real-time, multilingual, and assistive computer vision applications.


