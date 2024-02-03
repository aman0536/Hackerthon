import PyPDF2
import spacy
import requests
from transformers import pipeline

# Load the NLP model
nlp = pipeline("medical_report_analysis", model="nlm/bluebert-base-uncased")

# Function to extract text from PDF reports
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

# Function to analyze the medical report using NLP
def analyze_medical_report(report_text):
    result = nlp(report_text)
    return result

# Load the PDF report
pdf_path = "report.pdf"
report_text = extract_text_from_pdf(pdf_path)

# Analyze the medical report
analysis_result = analyze_medical_report(report_text)

# Print the analysis result
print(analysis_result)