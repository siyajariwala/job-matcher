# Resume ↔ Job Match Score

A small tool that measures how semantically similar a resume is to a job description — built as a hands-on exercise while learning how embeddings and cosine similarity actually work under the hood.

Upload a resume (PDF), paste a job description, and get a match score based on the *meaning* of the text — not just keyword overlap.

## Why I built this

This week I have been going deeper into embeddings, vector search, and cosine similarity. Rather than just reading theory, I wanted to build something small and real with it — this tool was that exercise. It's intentionally simple: no database, no auth, just the core concept working end to end.

## What it does

1. Extracts text from an uploaded resume PDF
2. Converts both the resume and job description into embeddings using a sentence-transformer model
3. Calculates cosine similarity between the two vectors — implemented from scratch, not from a library
4. Returns a match score with simple feedback (strong / moderate / weak match)

## Tech stack

- **Python**
- **sentence-transformers** (`all-MiniLM-L6-v2`) — converts text into embedding vectors
- **pdfplumber** — extracts text from resume PDFs
- **Streamlit** — simple web UI
- **Cosine similarity** — implemented manually with plain Python (`math` module only, no NumPy/scikit-learn dependency for the core calculation)

## How cosine similarity is calculated

Rather than calling a library function, the similarity score is computed directly from the formula:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

```python
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return dot_product / (magnitude_a * magnitude_b)
```

This was intentional — the goal was to understand the math, not just use a black-box call.

## Running it locally

```bash
# clone the repo
git clone https://github.com/siyajariwala/job-matcher.git
cd job-matcher

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install streamlit sentence-transformers pdfplumber

# run the app
streamlit run app.py
```

The app opens automatically in your browser. Upload a resume PDF, paste a job description, and click "Calculate Match Score."

## Limitations

- Embedding models have a token limit, so very long resumes or job descriptions may get truncated
- Match score reflects semantic similarity, not an exhaustive skills match — a high score doesn't guarantee qualification, and a moderate score doesn't mean a bad fit
- No document chunking yet, which would improve accuracy on longer text (a planned next step)

## What I'd improve next

- Add chunking so longer resumes/JDs are compared section by section instead of as one large block
- Extract and display specific matching vs. missing skills, not just an overall score
- Add support for multiple resume formats beyond PDF

## What I learned building this

- How raw text becomes a fixed-size embedding vector, and why models are trained rather than hand-coded
- Why cosine similarity — not Euclidean distance — is the right metric for comparing text embeddings (it measures direction/meaning, not length)
- The practical difference between cosine *similarity* and cosine *distance*, and why vector databases like ChromaDB return distance in their results
- How to extract text from PDFs using `pdfplumber`, and turn a terminal script into a usable Streamlit app
