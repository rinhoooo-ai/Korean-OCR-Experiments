# Korean OCR Project

**Author:** Phu Viet Tran, Phong Hoai Nguyen, Hung Dang Pham  
**Date:** October 12th 2025

## 1. Project Overview
This is a project from hackathon Chung's Innovation Challenge hosted in Ho Chi Minh City of Technology. It focuses on **Korean text recognition (OCR)** from dynamic images (GIF, WebP, JPG) using a full research pipeline:
1. **Preprocessing:** grayscale, gaussian blur, deskew, CLAHE, denoise, sharpen, and adaptive threshold.  
2. **OCR:** PaddleOCR (Korean language model).  
3. **Postprocessing:** Korean text segmentation using `konlpy` (Okt tokenizer) to improve text quality.  
*Note: Dataset is proprietary and part of a competition; only code and results are shared.*

## 2. Results

| Phase        | Accuracy (%) |
|--------------|--------------|
| Phase 1      | 76.0         |
| Phase 2      | 77.5         |
| Final        | 77.3         |

These results demonstrate the effectiveness of this pipeline.

## 3. Repository Structure
Korean-OCR-Experiments/
├─ notebooks/ 
│ ├─ ocr_experiments_raw.ipynb
│ └─ ocr_experiments_clean.ipynb
├─ src/
│ ├─ preprocessing.py
│ ├─ model.py
│ └─ postprocessing.py
├─ presentation.pptx # Presentation slides
├─ requirements.txt # Python dependencies
└─ README.md # Project description and instructions


## 4. Notes
- Some helper code and comments were assisted by ChatGPT, but the pipeline, models, preprocess ideas were come up with and tested by the authors.
- Dataset is private.
