# **Career Prediction through Resume Analysis**

The **Smart Resume Analyzer** is an advanced system designed to transform the resume evaluation process using Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL). This system automates the traditionally labor-intensive and subjective process of analyzing resumes, offering efficient and data-driven insights for both job seekers and recruiters.

## Aim  

The primary aim of this project is to automate and enhance the resume evaluation process by leveraging state-of-the-art technologies in Natural Language Processing, Machine Learning, and Graph Convolutional Networks. This tool is intended to assist job seekers in improving their resumes and enable recruiters to identify suitable candidates efficiently and accurately.

## Overview  

The **Smart Resume Analyzer** revolutionizes resume analysis by providing the following functionalities:  
- Automated parsing of resumes to extract essential details.  
- Job domain categorization using advanced classification models.  
- Sentiment and tone analysis of resumes.  
- Salary prediction based on experience and skills.  
- Career trajectory recommendations using Graph Convolutional Networks (GCNs).  

---

## Features  

- **Resume Parsing**: Extracts details like personal information, work experience, and skills from resumes.  
- **Job Role Categorization**: Uses text vectorization and classification models to categorize resumes.  
- **Tone Analysis**: Evaluates sentiment and professionalism of resume content.  
- **Salary Prediction**: Estimates salary expectations using a Random Forest Regressor.  
- **Career Path Modeling**: Provides career recommendations through advanced graph-based machine learning models.  

---

## Technologies Used  

- **Programming Language**: Python  
- **Frameworks**: Flask, Hugging Face Transformers  
- **Libraries**:  
  - Natural Language Processing: `nltk`, `spacy`, `transformers`  
  - Machine Learning: `scikit-learn`, `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Graph Models: `NetworkX`, `PyTorch Geometric`  

---

## Implementation  

### Resume Parsing  

- Uses **TF-IDF vectorization** to transform text data into numerical representations.  
- Extracts structured fields such as phone numbers, emails, and dates using regex.  

### Category Prediction  

- Employs **K-Nearest Neighbors (KNN)** classifier for job domain classification.  
- Utilizes **TF-IDF** for vectorization and PyMuPDF for data extraction.  

### Tone Prediction  

- Powered by **DistilBERT**, a lightweight transformer-based model.  
- Classifies tones into professional, neutral, or casual.  

### Salary Estimation  

- Uses a **Random Forest Regressor** for salary predictions.  
- Hyperparameter tuning ensures high accuracy.  

### Career Path Modeling  

- Implements **Graph Convolutional Networks (GCNs)** to represent relationships between technical skills, experience, and salary expectations.  
- Recommends optimal career trajectories based on structured graph data.  

---

# Conclusion 

The Smart Resume Analyser demonstrates the effectiveness of integrating advanced ML and NLP models for automating resume evaluation. It bridges the gap between subjective assessment and objective metrics, empowering both job seekers and recruiters. Future enhancements could include real-time feedback, expanded language support, and cloud deployment for scalability

 


