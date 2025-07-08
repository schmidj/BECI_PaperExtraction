import requests
from flask import Flask, render_template, request
from datetime import datetime
from xml.etree import ElementTree
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Initialize embedding model (load once)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for papers and their embeddings
PAPER_CACHE = None
EMBEDDINGS_CACHE = None
INDEX_CACHE = None


def fetch_arxiv_papers(query="marine heatwaves", max_results=20):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        authors = ', '.join([author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)])
        published = entry.find('atom:published', ns).text
        year = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").year
        journal_ref_elem = entry.find('atom:journal_ref', ns)
        journal = journal_ref_elem.text if journal_ref_elem is not None else "arXiv"
        link_elem = entry.find('atom:id', ns)
        url = link_elem.text if link_elem is not None else ""
        papers.append({
            'title': title,
            'authors': authors,
            'year': year,
            'journal': journal,
            'url': url
        })
    return papers


def get_papers_and_embeddings():
    global PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    if PAPER_CACHE is not None and EMBEDDINGS_CACHE is not None and INDEX_CACHE is not None:
        return PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    papers = fetch_arxiv_papers()
    texts = [paper['title'] + ". " + paper['authors'] + ". " + paper['journal'] for paper in papers]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    PAPER_CACHE = papers
    EMBEDDINGS_CACHE = embeddings
    INDEX_CACHE = index
    return papers, embeddings, index

@app.route('/')
def index():
    papers = fetch_arxiv_papers()
    return render_template('index.html', papers=papers)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return render_template('index.html', papers=[])
    papers, embeddings, index = get_papers_and_embeddings()
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, 10)  # top 10 results
    results = [papers[i] for i in I[0] if i < len(papers)]
    return render_template('index.html', papers=results)

if __name__ == '__main__':
    app.run(debug=True)
