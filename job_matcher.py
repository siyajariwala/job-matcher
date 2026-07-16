from sentence_transformers import SentenceTransformer
import math
import pdfplumber

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a,b):
    #Pair up a and b element by element, multiply each pair, then add all the products together
    dot_product = sum (x*y for x,y in zip(a,b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(y**2 for y in b))
    return dot_product / (magnitude_a * magnitude_b)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

job_description = """
PASTE THE FULL JOB DESCRIPTION HERE
"""

resume_path = input("Enter the path to your resume PDF: ")
resume = extract_text_from_pdf(resume_path)

# Ask user for job description
print("Paste the job description, then press Enter twice when done:")
job_description = ""
while True:
    line = input()
    if line == "":
        break
    job_description += line + " "
    
    
# take the job description and resume and encode them into embeddings using the embeddings model 
job_embedding = model.encode(job_description)
resume_embedding = model.encode(resume)



score = cosine_similarity(job_embedding, resume_embedding)

print(f"Match Score: {score * 100:.1f}%")
if score > 0.7:
    print("Strong match! 🎯")
elif score > 0.5:
    print("Moderate match — consider tailoring your resume")
else:
    print("Weak match — resume may need significant tailoring")