from sentence_transformers import SentenceTransformer
import math

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a,b):
    #Pair up a and b element by element, multiply each pair, then add all the products together
    dot_product = sum (x*y for x,y in zip(a,b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(y**2 for y in b))
    return dot_product / (magnitude_a * magnitude_b)

job_description = """
We are looking for a Software Engineer Intern with experience 
in Python, React, and building RAG pipelines using vector 
databases. Experience with FastAPI and LLM integration is a plus.
"""

resume = """
Built an end-to-end RAG pipeline using Python, FastAPI, and 
ChromaDB. Shipped production React components at a SaaS startup. 
Experience integrating Claude and Groq LLM APIs.
"""

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