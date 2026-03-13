import PyPDF2
import os
from dotenv import load_dotenv
from google import genai

#Load API key from .env file
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_file):
    """Uses PyPDF2 to extract text (skills, projects, certifications) from the resume."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_career_recommendations(job_role, resume_file):
    """Sends the extracted resume data and job role to the Gemini API."""
    resume_text = extract_text_from_pdf(resume_file)
    
    prompt = f"""
    You are an expert career counselor and technical mentor.
    
    I am providing you with the text extracted from a candidate's resume and their desired job role.
    Extract and analyze their current skills, projects, and certifications from the resume text.
    Then, compare it against the industry requirements for the role of: {job_role}.
    
    Provide a structured response strictly following these three sections:
    
    1. Roadmap for Missing Skills: Provide a learning roadmap for the domain/job-related skills the candidate is currently lacking.
    2. 3 Project Ideas: Suggest 3 specific, actionable project ideas to be implemented to build a strong portfolio for this role.
    3. Certification Programs: Recommend specific certification programs which can be done to bridge the gaps.
    
    ---
    Candidate Resume Text:
    {resume_text}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt
    )
    return response.text