import requests
from flask import Flask, render_template, request
from datetime import datetime
from xml.etree import ElementTree
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Initialize embedding model (load once)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for papers and their embeddings
PAPER_CACHE = None
EMBEDDINGS_CACHE = None
INDEX_CACHE = None


def fetch_arxiv_papers(query="marine heatwaves", max_results=1000):
    batch_size = 100  # arXiv API limit per request
    papers = []
    for start in range(0, max_results, batch_size):
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            authors = ', '.join([author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)])
            published = entry.find('atom:published', ns).text
            year = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").year
            journal_ref_elem = entry.find('atom:journal_ref', ns)
            journal = journal_ref_elem.text if journal_ref_elem is not None else "arXiv"
            link_elem = entry.find('atom:id', ns)
            url = link_elem.text if link_elem is not None else ""
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            # Try to extract keywords (if present)
            keywords_elem = entry.find('atom:keywords', ns)
            keywords = keywords_elem.text.lower() if keywords_elem is not None else ""
            # Filter for 'marine heatwave' or 'marine heatwaves' in title, abstract, or keywords
            if (
                'marine heatwave' in title.lower() or 'marine heatwaves' in title.lower() or
                'marine heatwave' in abstract.lower() or 'marine heatwaves' in abstract.lower() or
                'marine heatwave' in keywords or 'marine heatwaves' in keywords
            ):
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'url': url,
                    'abstract': abstract,
                    'source': 'arXiv',
                    'keywords': keywords
                })
        # Stop if fewer than batch_size results returned (end of results)
        if len(root.findall('atom:entry', ns)) < batch_size:
            break
    return papers

def fetch_semanticscholar_papers(query="marine heatwave", max_results=100):
    papers = []
    offset = 0
    batch_size = 100  # Semantic Scholar API max per request
    while len(papers) < max_results:
        params = {
            "query": query,
            "offset": offset,
            "limit": batch_size,
            "fields": "title,authors,year,venue,url,abstract"
        }
        response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        for paper in data.get('data', []):
            title = paper.get('title', '').strip()
            authors = ', '.join([a.get('name', '') for a in paper.get('authors', [])])
            year = paper.get('year', '')
            journal = paper.get('venue', 'Semantic Scholar')
            url = paper.get('url', '')
            abstract = paper.get('abstract', '')
            # Try to extract keywords (if present)
            keywords = ', '.join(paper.get('keywords', [])).lower() if 'keywords' in paper else ''
            # Filter for 'marine heatwave' or 'marine heatwaves' in title, abstract, or keywords
            if (
                'marine heatwave' in title.lower() or 'marine heatwaves' in title.lower() or
                'marine heatwave' in abstract.lower() or 'marine heatwaves' in abstract.lower() or
                'marine heatwave' in keywords or 'marine heatwaves' in keywords
            ):
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'url': url,
                    'abstract': abstract,
                    'source': 'Semantic Scholar',
                    'keywords': keywords
                })
        if len(data.get('data', [])) < batch_size:
            break
        offset += batch_size
    return papers

def get_papers_and_embeddings():
    global PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    if PAPER_CACHE is not None and EMBEDDINGS_CACHE is not None and INDEX_CACHE is not None:
        return PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    arxiv_papers = fetch_arxiv_papers()
    ss_papers = fetch_semanticscholar_papers()
    papers = arxiv_papers + ss_papers
    texts = [paper['title'] + ". " + paper['authors'] + ". " + paper['journal'] + ". " + paper['abstract'] for paper in papers]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    PAPER_CACHE = papers
    EMBEDDINGS_CACHE = embeddings
    INDEX_CACHE = index
    return papers, embeddings, index

@app.route('/')
def index():
    arxiv_papers = fetch_arxiv_papers()
    ss_papers = fetch_semanticscholar_papers()
    papers = arxiv_papers + ss_papers
    # Sort by year descending, then by source
    papers = sorted(papers, key=lambda x: (x['year'] if x['year'] else 0), reverse=True)
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
