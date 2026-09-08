import os, sys, time, hashlib, re, uuid, json, pickle
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup, Tag

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# ── CONFIGURATION & CONSTANTS ────────────────────────────────────────────────
TARGET_URL    = "https://docs.python.org/3/"
MAX_PAGES     = 30
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
CACHE_FILE    = "rag_v4_cache.pkl"
CHROMA_DIR    = "./chroma_v4_base"

HEADERS = {"User-Agent": "MyAdviceRAGAgent/Production"}

# ── 1. CRAWLER ───────────────────────────────────────────────────────────────
def is_valid_url(url, base_netloc, base_prefix):
    p = urlparse(url)
    if p.scheme not in ("http","https"): return False
    if p.netloc != base_netloc: return False
    if not p.path.startswith(base_prefix): return False
    bad = (".pdf",".zip",".tar",".gz",".png",".jpg",".jpeg",".gif",".css",".js",".xml",".svg")
    return not any(p.path.lower().endswith(e) for e in bad)

def crawl_website(start_url, max_pages=30):
    ps = urlparse(start_url)
    base_netloc, base_prefix = ps.netloc, ps.path
    queue, visited, pages = deque([start_url]), set(), []
    print(f"\\n🕸️  [INGESTION JOB] Crawling {start_url} (max {max_pages} pages)")
    
    while queue and len(pages) < max_pages:
        url = urldefrag(queue.popleft())[0]
        if url in visited: continue
        if not is_valid_url(url, base_netloc, base_prefix): continue
        visited.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            if "text/html" not in r.headers.get("Content-Type",""): continue
            pages.append({
                "url": url, 
                "html": r.text,
                "content_hash": hashlib.md5(r.text.encode()).hexdigest()
            })
            print(f"  [{len(pages):>3}/{max_pages}]  Downloaded: {url}")
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                if is_valid_url(nxt, base_netloc, base_prefix) and nxt not in visited:
                    queue.append(nxt)
            time.sleep(0.15)
        except requests.RequestException as e:
            print(f"  ⚠️  Skipping {url} -> {e}")
            
    print(f"✅ Crawled {len(pages)} pages successfully.")
    return pages

# ── 2. EXTRACTION ─────────────────────────────────────────────────────────────
CRAWL_TS = datetime.now(timezone.utc).isoformat()

def get_page_title(soup):
    h1 = soup.find("h1")
    if h1: return h1.get_text(strip=True)
    t = soup.find("title")
    return t.get_text(strip=True) if t else "Untitled"

def extract_sections(html, url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","nav","footer","header","noscript","aside","form"]):
        tag.decompose()
    page_title = get_page_title(soup)
    main = (soup.find("div", {"role":"main"}) or soup.find("main") or soup.find("article") or soup.body)
    if not main: return []

    sections = []
    current_heading = page_title
    current_texts   = []

    for elem in main.descendants:
        if not isinstance(elem, Tag): continue
        if elem.name in {"h1","h2","h3"}:
            body = "\\n".join(t.strip() for t in current_texts if t.strip())
            if body:
                sections.append({"title": page_title, "section": current_heading, "text": body})
            current_heading = elem.get_text(strip=True)
            current_texts   = []
        elif elem.name in ("p","li","pre","code","dd","dt","td","th"):
            txt = elem.get_text(separator=" ", strip=True)
            if txt:
                current_texts.append(txt)

    body = "\\n".join(t.strip() for t in current_texts if t.strip())
    if body:
        sections.append({"title": page_title, "section": current_heading, "text": body})

    return sections

def build_documents(pages):
    docs = []
    for page in pages:
        url     = page["url"]
        doc_id  = hashlib.md5(url.encode()).hexdigest()[:12]
        sections = extract_sections(page["html"], url)
        for sec in sections:
            if len(sec["text"]) < 80: continue
            enriched = f"[{sec['title']} > {sec['section']}]\\n{sec['text']}"
            docs.append(Document(
                page_content=enriched,
                metadata={
                    "source":          url,
                    "title":           sec["title"],
                    "section":         sec["section"],
                    "doc_id":          doc_id,
                    "crawl_timestamp": CRAWL_TS,
                    "content_hash":    page["content_hash"],
                }
            ))
    return docs

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_\\.]+", text.lower())

# ── 3. JOB EXECUTION ──────────────────────────────────────────────────────────
def run_ingestion_pipeline():
    print("🚀 Starting offline ingestion pipeline...")
    
    pages = crawl_website(TARGET_URL, max_pages=MAX_PAGES)
    documents = build_documents(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\\n\\n", "\\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents)

    chunks = []
    for i, c in enumerate(raw_chunks):
        new_meta = dict(c.metadata)
        new_meta["chunk_id"] = f"{new_meta['doc_id']}-{i}"
        chunks.append(Document(page_content=c.page_content, metadata=new_meta))

    print(f"✅ Created {len(chunks)} contextual chunks.")

    print("\\n⏳ Loading BGE-base embedding model on CPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )

    print("⏳ Building ChromaDB vector store... (This takes a few minutes, writing to disk)")
    t0 = time.time()
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = "rag_v4_kb_base",
        persist_directory = CHROMA_DIR,
    )
    print(f"✅ ChromaDB written to {CHROMA_DIR} ({time.time()-t0:.1f}s)")

    print("⏳ Extracting corpus tokens for BM25 Keyword Search...")
    corpus_tokens = [tokenize(c.page_content) for c in chunks]
    bm25_index    = BM25Okapi(corpus_tokens)
    print(f"✅ BM25 index built.")
    
    print("\\n💾 Serializing sparse indexes to disk for instant query loads...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "bm25": bm25_index}, f)
    print(f"✅ Cache saved successfully to {CACHE_FILE}!")
    print("🎉 INGESTION PIPELINE COMPLETE. You can now run the query server instantly.")

if __name__ == "__main__":
    run_ingestion_pipeline()
