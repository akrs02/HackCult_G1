import os
from flask import Flask, render_template, request, jsonify
import re
from pdfminer.high_level import extract_text
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('stopwords')

app = Flask(__name__)

# Define paths for storing uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure that the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Function to extract emails
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)

# Function to extract skills    
def extract_skills(input_text, job_details):
    found_skills = set()
    SKILLS_DB = job_details['skills']  # Get the skills from the job description
    for skill in SKILLS_DB:
        # Use word boundaries to match whole words
        pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
        if pattern.search(input_text):
            found_skills.add(skill)
    return found_skills

# Function to extract education details
def extract_education(resume_text):
    college_pattern = re.compile(r'Siddaganga Institute Of Technology')
    gpa_pattern1 = re.compile(r'CGPA:\s*(\d+(\.\d+)?)(?:/\d+)?')  
    gpa_pattern2 = re.compile(r'(\d+(\.\d+)?)(?:/\d+)?\s* CGPA')
    college = college_pattern.search(resume_text)
    gpa = gpa_pattern1.search(resume_text) or gpa_pattern2.search(resume_text)
    return {'gpa': gpa.group(1) if gpa else None}

# Function to extract name
def extract_name(resume_text):
    lines = resume_text.splitlines()
    for line in lines:
        if re.match(r'^[A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*', line):
            return line.strip() 
    return None

def extract_experience(text):
    pattern = r'(.*?)\n(.*?)\s–\s(.*?)\n([A-Za-z]+\s\d{4})\s–\s([A-Za-z]+\s\d{4}|Present)'
    matches = re.findall(pattern, text)
    return matches

def calculate_years_of_experience(experience):
    total_years = 0
    for job in experience:
        position, company, address, start, end = job
        start_date = datetime.strptime(start, "%B %Y")
        if end == "Present":
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end, "%B %Y")
        years = (end_date - start_date).days / 365.25
        total_years += years
    return total_years

# Function to parse the resume and return the extracted data
def parse_resume(resume_path, job_details):
    text = extract_text_from_pdf(resume_path)
    name = extract_name(text)
    email = extract_emails(text)
    extracted_experience = extract_experience(text)
    experience = calculate_years_of_experience(extracted_experience)
    
    if experience < float(job_details["qualifications"][1][0]):
        experience = None
        
    skills = extract_skills(text, job_details)
    education = extract_education(text)
    
    return {
        'name': name,
        'email': email[0] if email else None,
        'experience': experience,
        'skills': list(skills),
        'education': education
    }

# Function to extract job details from the job description
def extract_job_details(job_description_text):
    skills_pattern = r'(?<=Skills:)(.*?)(?=Qualifications:|Responsibilities:|Benefits:|$)'
    responsibilities_pattern = r'(?<=Responsibilities:)(.*?)(?=Qualifications:|Benefits:|$)'
    qualifications_pattern = r'(?<=Qualifications:)(.*?)(?=Responsibilities:|Benefits:|$)'

    skills_match = re.search(skills_pattern, job_description_text, re.DOTALL)
    skills = []
    if skills_match:
        skills_text = skills_match.group(0)
        skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]

    responsibilities_match = re.search(responsibilities_pattern, job_description_text, re.DOTALL)
    responsibilities = []
    if responsibilities_match:
        responsibilities_text = responsibilities_match.group(0)
        responsibilities = [responsibility.strip() for responsibility in responsibilities_text.split('\n') if responsibility.strip()]

    qualifications_match = re.search(qualifications_pattern, job_description_text, re.DOTALL)
    qualifications = []
    if qualifications_match:
        qualifications_text = qualifications_match.group(0)
        qualifications = [qualification.strip() for qualification in qualifications_text.split('\n') if qualification.strip()]

    job_details = {
        "skills": skills,
        "responsibilities": responsibilities,
        "qualifications": qualifications
    }

    return job_details

# Web route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Web route to handle file upload and processing
@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({'error': 'Both files are required!'}), 400
    
    resume_file = request.files['resume']
    job_description_file = request.files['job_description']
    
    if resume_file.filename == '' or job_description_file.filename == '':
        return jsonify({'error': 'Both files are required!'}), 400

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resume.pdf')
    job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_description.txt')
    
    resume_file.save(resume_path)
    job_description_file.save(job_description_path)

    with open(job_description_path, 'r') as file:
        job_description_text = file.read()

    job_details = extract_job_details(job_description_text)
    parsed_data = parse_resume(resume_path, job_details)

    intersecting_skills = list(set(parsed_data["skills"]).intersection(set(job_details["skills"])))
    missing_skills = list(set(job_details["skills"]) - set(parsed_data["skills"]))

    processed_job_skills = " ".join(job_details["skills"])
   
    # Generate similarity score
    vectorizer = TfidfVectorizer()
    intersecting_skills_str = " ".join(intersecting_skills)
    tfidf_matrix = vectorizer.fit_transform([intersecting_skills_str, processed_job_skills])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = cosine_sim[0][0]
    qualified='Qualified' if similarity_score>=0.65 else 'Not Qualified'
    return jsonify({
        'name': parsed_data['name'],
        'email': parsed_data['email'],
        'matched_skills': intersecting_skills,
        'similarity_score': similarity_score,
        'missing_skills': missing_skills,
        'qualified':qualified
    })

if __name__ == '__main__':
    app.run(debug=True, port=3000)