

from flask import Flask, render_template, request
import fitz
import os
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import GROQ_API_KEY

app = Flask(__name__)

UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_analysis={}
latest_resume_text=""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global latest_analysis
    global latest_resume_text

    resume = request.files.get("resume")
    job_description = request.form.get("job_description", "")

    # No file selected
    if resume is None or resume.filename == "":
        return "<h3>❌ Please select a PDF resume to upload.</h3>"

    # Only allow PDF files
    if not resume.filename.lower().endswith(".pdf"):
        return "<h3>❌ Only PDF files are supported.</h3>"

    path = os.path.join(UPLOAD_FOLDER, resume.filename)
    resume.save(path)

    # Open PDF safely
    try:
        pdf = fitz.open(path)
    except Exception:
        return "<h3>❌ Unable to open the PDF. The file may be corrupted.</h3>"

    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()

    # No readable text
    if not text.strip():
        return "<h3>❌ No readable text was found in the PDF. Please upload a text-based resume.</h3>"

    global latest_resume_text
    latest_resume_text=text

    resume_match = 0
    matched_skills = []
    missing_skills_jd = []

    if job_description.strip() != "":
        documents = [text, job_description]

        vectorizer = CountVectorizer(stop_words="english")

        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(matrix)[0][1]

        resume_match = round(similarity * 100)

    # -----------------------------
    # ATS Keyword Matching
    # -----------------------------

        common_skills = [
            "python","java","c","c++","sql","mysql","html","css","javascript",
            "flask","django","react","node","mongodb","oracle","aws","azure",
            "docker","kubernetes","git","github","linux","selenium","devops",
            "machine learning","deep learning","tensorflow","keras","pandas",
            "numpy","scikit-learn","opencv","nlp","power bi","tableau",
            "excel","jira","postman","rest api","spring","spring boot",
            "firebase","arduino","iot","networking","cybersecurity"
        ]

        resume_lower = text.lower()
        jd_lower = job_description.lower()

        for skill in common_skills:
            
            if skill in jd_lower:
                
                if skill in resume_lower:
                    matched_skills.append(skill.title())
                else:
                    missing_skills_jd.append(skill.title())

    prompt=f"""
You are an experienced Technical Recruiter.
Return ONLY valid JSON with keys:
resume_score, ats_score, strengths, weaknesses,
missing_skills, project_feedback,interview_readiness,
suggestions, recommended_role, role_reason.

Based on the candidate's skills, projects, certifications, internship and education,
recommend ONLY ONE most suitable career role.

Examples:
Python Developer
Java Developer
Cybersecurity Analyst
Data Analyst
Machine Learning Engineer
Web Developer
Software Developer
DevOps Engineer
Cloud Engineer

Also provide one short reason (2–3 sentences) explaining why this role best matches the resume.

Resume:
{text}
"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            },
            timeout=60
        )

        result = r.json()

        if "error" in result:
            return "<h3>❌ AI service is temporarily unavailable. Please try again later.</h3>"
        if "choices" not in result:
            return "<h3>❌ Unable to process the AI response. Please try again.</h3>"

        analysis = json.loads(result["choices"][0]["message"]["content"])

        analysis.setdefault(
            "recommended_role",
            "Software Developer"
        )

        analysis.setdefault(
            "role_reason",
            "This role best matches the candidate's technical skills and projects."
        )

    except requests.exceptions.Timeout:
        return "<h3>❌ AI request timed out. Please try again.</h3>"

    except requests.exceptions.ConnectionError:
        return "<h3>❌ Unable to connect to the AI service. Check your internet connection.</h3>"

    except requests.exceptions.RequestException:
        return "<h3>❌ AI service is currently unavailable. Please try again later.</h3>"

    except json.JSONDecodeError:
        return "<h3>❌ AI returned an invalid response. Please try again.</h3>"

    except Exception:
        return "<h3>❌ Something went wrong while analyzing your resume.</h3>"

    analysis["resume_match"]=resume_match
    analysis["matched_skills"] = matched_skills
    analysis["missing_skills_jd"] = missing_skills_jd
    latest_analysis=analysis
    latest_resume_text = text

    return render_template("result.html",analysis=analysis)

@app.route("/interview_questions")
def interview_questions():

    global latest_resume_text

    prompt = f"""
You are a Senior Technical Interviewer.

Based on the resume below, generate interview questions.

IMPORTANT RULES:

1. Return ONLY valid HTML.

2. Do NOT use markdown.

3. Do NOT use tables.

4. Do NOT use **

5. Create exactly THREE sections.

Section 1:
<h2>📘 Technical Skills</h2>

<ul>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
</ul>

Section 2:
<h2>📂 Projects</h2>

<ul>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
</ul>

Section 3:
<h2>👨‍💼 HR Questions</h2>

<ul>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
<li>Question</li>
</ul>

Generate ONLY the HTML.

Resume:

{latest_resume_text}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()

    if "error" in result:
        return f"<pre>{result['error']}</pre>"

    questions = result["choices"][0]["message"]["content"]

    return render_template(
        "interview.html",
        questions=questions
    )
@app.route("/resume_summary")
def resume_summary():

    global latest_resume_text

    prompt = f"""
You are an expert Resume Writer.

Based on the resume below,

write ONE professional ATS-friendly resume summary.

Rules:

1. Maximum 5 lines.
2. Professional language.
3. Mention important technical skills.
4. Mention strengths.
5. Do NOT invent information.
6. Return ONLY the summary.
7. Do NOT use markdown.
8. Do NOT use headings.

Resume:

{latest_resume_text}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()

    if "error" in result:
        return f"<pre>{result['error']}</pre>"

    summary = result["choices"][0]["message"]["content"]

    return render_template(
        "resume_summary.html",
        summary=summary
    )

@app.route("/cover_letter")
def cover_letter():

    global latest_resume_text

    prompt = f"""
You are an expert HR Recruiter.

Based on the resume below, write a professional cover letter.

Rules:

1. Maximum 300 words.
2. Professional and formal tone.
3. Highlight the candidate's skills and strengths.
4. Do NOT invent experience.
5. Do NOT invent projects.
6. Do NOT invent certifications.
7. Return ONLY the cover letter.
8. Do NOT use markdown.

Resume:

{latest_resume_text}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()

    if "error" in result:
        return f"<pre>{result['error']}</pre>"

    letter = result["choices"][0]["message"]["content"]

    return render_template(
        "cover_letter.html",
        letter=letter
    )

@app.route("/interview_answers")
def interview_answers():

    global latest_resume_text
    
    prompt = f"""
You are an experienced Technical Interviewer.

Based on the resume below, generate interview questions with professional sample answers.

Rules:

1. Generate exactly 5 questions.
2. Each question must have one detailed sample answer.
3. Include HR, Technical, and Project questions.
4. Do NOT use markdown.
5. Do NOT use tables.
6. Keep answers between 40 and 60 words.
7. Return ONLY plain text.
8. Use ONLY the information present in the resume.
9. Do NOT invent:
- work experience
- projects
- technologies
- achievements
10. If the resume belongs to a fresher, generate fresher-level interview questions.
11. Every question must be directly related to the candidate's skills, internship, certifications, or projects.

Format exactly like this:

Question:
...

Answer:
...

Question:
...

Answer:
...

Resume:

{latest_resume_text}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "reasoning_effort": "low",
        "max_completion_tokens": 1800
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()

    
    if "error" in result:
        return f"<pre>{result['error']}</pre>"
    if "choices" not in result:
        return f"<pre>{result}</pre>"

    answers = result["choices"][0]["message"]["content"]

    return render_template(
        "interview_answers.html",
        answers=answers
    )

@app.route("/dashboard")
def dashboard():

    global latest_analysis

    return render_template(
        "dashboard.html",
        analysis=latest_analysis
    )

if __name__=="__main__":
    app.run(debug=True)
