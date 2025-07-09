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
OPENALEX_API_URL = "https://api.openalex.org/works"

# Initialize embedding model (load once)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for papers and their embeddings
PAPER_CACHE = None
EMBEDDINGS_CACHE = None
INDEX_CACHE = None

# Add keyword lists for categorization
KEYWORDS_ECOLOGICAL_IMPACTS = [
    "ecology", "ecosystem", "species", "biodiversity", "habitat",
    "mortality", "coral bleaching", "seagrass", "kelp", "trophic",
    "fisheries", "population decline", "community shift"
]

KEYWORDS_PHYSICAL_OCEANOGRAPHY = [
    "SST", "sea surface temperature", "stratification", "upwelling",
    "ENSO", "El Niño", "ocean circulation", "heat flux",
    "atmospheric forcing", "thermal anomaly", "climate variability"
]

KEYWORDS_MODELING_AND_FORECASTING = [
    "model", "simulation", "forecast", "prediction", "statistical model",
    "machine learning", "downscaling", "ensemble", "projection",
    "scenario", "hindcast"
]

KEYWORDS_SOCIOECONOMIC_IMPACTS = [
    "fisheries", "aquaculture", "economic loss", "livelihoods",
    "tourism", "food security", "policy", "coastal communities",
    "management strategies"
]

KEYWORDS_BIOGEOCHEMISTRY = [
    "oxygen", "hypoxia", "nutrient", "chlorophyll", "carbon cycle",
    "acidification", "primary productivity", "nitrogen"
]

KEYWORDS_DETECTION_AND_METRICS = [
    "detection", "metric", "definition", "threshold", "heatwave event",
    "marine heatwave index", "climatology", "duration", "intensity",
    "anomaly"
]

KEYWORDS_RESILIENCE_AND_ADAPTATION = [
    "resilience", "adaptation", "acclimation", "recovery", "tolerance",
    "phenotypic plasticity", "evolutionary response", "resistance"
]
MAILTO = "jul.s.schmid@gmail.com"

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
            keywords_elem = entry.find('atom:keywords', ns)
            keywords = keywords_elem.text.lower() if keywords_elem is not None else ""
            doi_elem = entry.find('atom:doi', ns)
            doi = doi_elem.text.lower() if doi_elem is not None else ""
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
                    'keywords': keywords,
                    'doi': doi
                })
        if len(root.findall('atom:entry', ns)) < batch_size:
            break
    return papers

def fetch_semanticscholar_papers(query="marine heatwave", max_results=100):
    papers = []
    offset = 0
    batch_size = 100  # Semantic Scholar API max per request
    try:
        while len(papers) < max_results:
            params = {
                "query": query,
                "offset": offset,
                "limit": batch_size,
                "fields": "title,authors,year,venue,url,abstract,keywords,externalIds"
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
                keywords = ', '.join(paper.get('keywords', [])).lower() if 'keywords' in paper else ''
                doi = paper.get('externalIds', {}).get('DOI', '').lower() if 'externalIds' in paper else ''
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
                        'keywords': keywords,
                        'doi': doi
                    })
            if len(data.get('data', [])) < batch_size:
                break
            offset += batch_size
    except requests.exceptions.HTTPError as e:
        print(f"Warning: Semantic Scholar API unavailable or rate limited: {e}")
        return []
    return papers

def classify_paper(title, abstract):
    text = (title + " " + (abstract or "")).lower()
    categories = []

    if any(k.lower() in text for k in KEYWORDS_ECOLOGICAL_IMPACTS):
        categories.append("ecological impacts")
    if any(k.lower() in text for k in KEYWORDS_PHYSICAL_OCEANOGRAPHY):
        categories.append("physical oceanography")
    if any(k.lower() in text for k in KEYWORDS_MODELING_AND_FORECASTING):
        categories.append("modeling and forecasting")
    if any(k.lower() in text for k in KEYWORDS_SOCIOECONOMIC_IMPACTS):
        categories.append("socioeconomic impacts")
    if any(k.lower() in text for k in KEYWORDS_BIOGEOCHEMISTRY):
        categories.append("biogeochemistry")
    if any(k.lower() in text for k in KEYWORDS_DETECTION_AND_METRICS):
        categories.append("detection and metrics")
    if any(k.lower() in text for k in KEYWORDS_RESILIENCE_AND_ADAPTATION):
        categories.append("resilience and adaptation")
    
    # If no specific categories found, mark as "other"
    if not categories:
        categories.append("other")
    
    return categories

def fetch_openalex_papers(query="marine heatwave", max_results=100):
    papers = []
    per_page = 25  # OpenAlex API max per page for this approach
    max_pages = max_results // per_page
    for page in range(1, max_pages + 1):
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "mailto": MAILTO
        }
        response = requests.get(OPENALEX_API_URL, params=params)
        if response.status_code != 200:
            print(f"OpenAlex failed at page {page}: {response.status_code}")
            break
        data = response.json()
        for result in data.get('results', []):
            title = result.get('title', '').strip()
            authors = ', '.join([a.get('author', {}).get('display_name', '') for a in result.get('authorships', [])])
            year = result.get('publication_year', '')
            # Try to get journal from host_venue.display_name, fallback to fetching full work details if missing or generic
            journal = result.get('host_venue', {}).get('display_name', '')
            url = result.get('primary_location', {}).get('source', {}).get('url', result.get('id', ''))
            abstract = result.get('abstract_inverted_index', {})
            if abstract:
                abstract_words = sorted([(pos, word) for word, poses in abstract.items() for pos in poses])
                abstract = ' '.join([word for pos, word in abstract_words])
            else:
                abstract = ''
            keywords = ', '.join([c.get('display_name', '') for c in result.get('concepts', [])]).lower()
            doi = result.get('doi', '').lower()
            # If journal is missing or generic, fetch full work details
            if not journal or journal.lower() in ["openalex", "", None]:
                work_id = result.get('id', '').replace('https://openalex.org/', '')
                if work_id:
                    work_url = f"https://api.openalex.org/works/{work_id}"
                    work_resp = requests.get(work_url)
                    if work_resp.status_code == 200:
                        work_data = work_resp.json()
                        # Prefer primary_location.source.display_name, fallback to host_venue.display_name
                        journal = work_data.get('primary_location', {}).get('source', {}).get('display_name', '')
                        if not journal:
                            journal = work_data.get('host_venue', {}).get('display_name', '')
            if (
                'marine heatwave' in title.lower() or 'marine heatwaves' in title.lower() or
                'marine heatwave' in abstract.lower() or 'marine heatwaves' in abstract.lower() or
                'marine heatwave' in keywords or 'marine heatwaves' in keywords
            ):
                categories = classify_paper(title, abstract)
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'url': url,
                    'abstract': abstract,
                    'source': 'OpenAlex',
                    'keywords': keywords,
                    'doi': doi,
                    'categories': categories
                })
    return papers

def deduplicate_papers(papers):
    seen_dois = set()
    unique_papers = []
    for paper in papers:
        doi = paper.get('doi', '').lower()
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_papers.append(paper)
        elif not doi:
            unique_papers.append(paper)
    return unique_papers

def get_papers_and_embeddings():
    global PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    if PAPER_CACHE is not None and EMBEDDINGS_CACHE is not None and INDEX_CACHE is not None:
        return PAPER_CACHE, EMBEDDINGS_CACHE, INDEX_CACHE
    arxiv_papers = fetch_arxiv_papers()
    ss_papers = fetch_semanticscholar_papers()
    openalex_papers = fetch_openalex_papers()
    papers = arxiv_papers + ss_papers + openalex_papers
    papers = deduplicate_papers(papers)
    # Add category to all papers (arXiv and Semantic Scholar)
    for paper in papers:
        if 'categories' not in paper:
            paper['categories'] = classify_paper(paper.get('title', ''), paper.get('abstract', ''))
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
    category_filter = request.args.get('category', '')
    query_filter = request.args.get('query', '')
    
    arxiv_papers = fetch_arxiv_papers()
    ss_papers = fetch_semanticscholar_papers()
    openalex_papers = fetch_openalex_papers()
    papers = arxiv_papers + ss_papers + openalex_papers
    papers = deduplicate_papers(papers)
    for paper in papers:
        if 'categories' not in paper:
            paper['categories'] = classify_paper(paper.get('title', ''), paper.get('abstract', ''))
    
    # Apply category filter if specified
    if category_filter:
        papers = [paper for paper in papers if any(c.lower() in category_filter.lower() for c in paper.get('categories', []))]
    
    # Apply semantic search if query is provided
    if query_filter:
        papers, embeddings, index = get_papers_and_embeddings()
        query_emb = model.encode([query_filter], convert_to_numpy=True)
        D, I = index.search(query_emb, 50)  # top 50 results
        search_results = [papers[i] for i in I[0] if i < len(papers)]
        # Apply category filter to search results if specified
        if category_filter:
            search_results = [paper for paper in search_results if any(c.lower() in category_filter.lower() for c in paper.get('categories', []))]
        papers = search_results
    
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
