# Gen AI Developer Roadmap
## From Data Scientist to Gen AI Developer
### 8-Week Project Plan with NVIDIA GPU

---

## Project Overview

This roadmap guides you through building a production-ready cross-domain **Content Discovery system** using RAG (Retrieval-Augmented Generation) with multi-agent capabilities. You'll leverage your NVIDIA RTX 5080 GPU to run local LLMs while building a system that recommends movies, books, AND podcasts based on themes, mood, and style.

This demonstrates essential Gen AI engineering skills:
- Cross-domain semantic search
- Multi-modal embeddings
- Agent orchestration
- Production deployment
- Sophisticated technical depth

## What You'll Build

**Cross-Domain Content Discovery System**: A multi-agent RAG application that finds content across movies, books, and podcasts based on natural language queries like:
- "I loved the book Sapiens, find me podcasts and documentaries with similar themes"
- "Show me sci-fi content across all formats that explores AI ethics"
- "Books like the movie Interstellar"

The system demonstrates:
- Cross-domain reasoning
- Hybrid search
- Conversation context
- Explainable recommendations

---

## GPU Capabilities (NVIDIA RTX 5080)

| Model | Parameters | Memory Required | Performance |
|-------|-----------|-----------------|-------------|
| Llama 3.2/3.3 | 8B | ~6-8GB | Excellent |
| Mistral 7B | 7B | ~5-7GB | Excellent |
| Qwen 2.5 | 7B-14B | ~5-12GB | Very Good |
| Phi-4 | 14B | ~10-12GB | Good |

---

## Free Content Data Sources

All data for this project is freely available:

### Movies
- **TMDb API** - themoviedb.org/documentation/api
- **IMDb Datasets** - developer.imdb.com/non-commercial-datasets
- **Wikipedia** - Movie pages with plot descriptions

### Books
- **Google Books API** - Free API access
- **Open Library API** - openlibrary.org
- **Goodreads datasets** - Available via Kaggle

### Podcasts
- **ListenNotes API** - Free tier available
- **iTunes Search API** - Free access
- **Spotify Podcast API** - Free tier

**Target**: ~1,000-2,000 items per content type for meaningful recommendations

---

## Phase 1: Environment Setup
### Week 1 • Days 1-7

### Days 1-2: Install Core Tools

```bash
# Install Ollama (easiest way to run local LLMs)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2:8b
ollama pull mistral:7b

# Test it
ollama run llama3.2:8b
```

### Days 3-4: Python Environment

```bash
# Create virtual environment
python -m venv genai_env
source genai_env/bin/activate  # Windows: genai_env\Scripts\activate

# Install essential packages
pip install langchain langchain-community
pip install chromadb sentence-transformers
pip install pypdf gradio ollama
```

### Days 5-7: GPU Verification & Basics

- [ ] Verify CUDA installation with `nvidia-smi`
- [ ] Test sentence-transformers with GPU acceleration
- [ ] Build simple 'Hello World' with Ollama
- [ ] Benchmark inference speeds on your hardware

---

## Phase 2: Build Basic RAG System
### Weeks 2-3

### Week 2: Document Processing & Embedding

**Tasks:**
- [ ] Set up API keys: TMDb, Google Books, ListenNotes/iTunes (all free tiers)
- [ ] Collect Movies: plot summaries, genres, themes (~1,000-2,000 movies)
- [ ] Collect Books: descriptions, genres, themes, author info (~1,000-2,000 books)
- [ ] Collect Podcasts: episode descriptions, topics, hosts (~500-1,000 podcasts/episodes)
- [ ] Design unified schema: title, description, themes, content_type, metadata
- [ ] Implement chunking strategies for descriptions across content types
- [ ] Generate embeddings using sentence-transformers (GPU-accelerated)
- [ ] Store in ChromaDB with content_type filtering and cross-domain search

**Code Example: Cross-Domain Data Processing**

```python
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Unified content schema
def create_content_doc(item, content_type):
    doc = f"Title: {item['title']}\n"
    doc += f"Type: {content_type}\n"
    doc += f"Description: {item['description']}\n"
    doc += f"Themes: {', '.join(item.get('themes', []))}\n"
    return doc

# Collect from multiple sources
movies = fetch_movies_from_tmdb()  # TMDb API
books = fetch_books_from_google()  # Google Books API
podcasts = fetch_podcasts()        # ListenNotes API

# Process all content types
all_content = []
for movie in movies:
    all_content.append({
        'text': create_content_doc(movie, 'movie'),
        'metadata': {'type': 'movie', 'year': movie['year']}
    })

# Generate embeddings and store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_texts(
    [c['text'] for c in all_content],
    embeddings,
    metadatas=[c['metadata'] for c in all_content]
)
```

### Week 3: Query & Retrieval

**Tasks:**
- [ ] Build semantic search across all content types
- [ ] Add content_type filtering (movies only, books only, or mixed)
- [ ] Implement cross-domain queries: 'books like the movie Interstellar'
- [ ] Test theme-based retrieval: 'existential content about AI'
- [ ] Connect retrieval to Ollama LLM for natural language responses
- [ ] Create diverse test queries across domains
- [ ] Measure cross-domain retrieval quality and relevance
- [ ] Add conversation memory: 'show me more like that but in podcast form'

**Code Example: Cross-Domain Recommendation Chain**

```python
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

llm = Ollama(model="llama3.2:8b")

# Create retrieval chain with cross-domain support
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={
        "k": 10,  # Get more results for diversity
        "filter": None  # Search across all content types
    }),
    return_source_documents=True
)

# Cross-domain query examples
query = "I loved the book Sapiens. Find me documentaries and podcasts with similar themes about human evolution and society."

result = qa_chain({"query": query})

# Show recommendations across content types
print(result["result"])
for doc in result["source_documents"]:
    content_type = doc.metadata['type']
    print(f"{content_type.upper()}: {doc.metadata['title']}")
```

---

## Phase 3: Add Multi-Agent Layer
### Weeks 4-5

### Week 4: Agent Framework Setup

**Tasks:**
- [ ] Install CrewAI or AutoGen framework
- [ ] Design your agent system architecture
- [ ] Define agent roles (see below)
- [ ] Plan agent interactions and workflows
- [ ] Study agent communication patterns

**Agent Architecture:**

- **Discovery Agent**: Searches across movies, books, and podcasts for thematic matches
- **Cross-Domain Reasoning Agent**: Identifies common themes across different content formats
- **Curator Agent**: Balances recommendations across content types and ensures diversity
- **Theme Analyzer Agent**: Extracts and matches deep thematic elements (not just keywords)
- **Explainer Agent**: Articulates why content across different formats shares themes
- **Orchestrator**: Coordinates multi-step cross-domain searches and manages context

### Week 5: Implementation & Testing

```python
from crewai import Agent, Task, Crew

discovery_agent = Agent(
    role='Cross-Domain Content Discovery',
    goal='Find thematically similar content across movies, books, podcasts',
    backstory='Expert at identifying common themes across different media formats',
    llm=ollama_llm
)

theme_analyzer = Agent(
    role='Theme Analyzer',
    goal='Extract and match deep thematic elements beyond surface keywords',
    backstory='Understands narrative patterns, philosophical themes, and cultural contexts',
    llm=ollama_llm
)

curator_agent = Agent(
    role='Multi-Format Curator',
    goal='Balance recommendations across content types',
    backstory='Ensures diverse, high-quality content mix',
    llm=ollama_llm
)

explainer_agent = Agent(
    role='Cross-Domain Explainer',
    goal='Articulate thematic connections across different formats',
    backstory='Expert at explaining why a book and podcast share themes',
    llm=ollama_llm
)

# Coordinate cross-domain discovery workflow
crew = Crew(agents=[discovery_agent, theme_analyzer, curator_agent, explainer_agent])
```

**Tasks:**
- [ ] Start with simple single-step queries
- [ ] Progress to complex multi-step research tasks
- [ ] Compare multi-agent vs single-agent performance
- [ ] Debug agent collaboration issues
- [ ] Optimize agent prompts and workflows

---

## Phase 4: Enhancement & Optimization
### Week 6

### Advanced RAG Techniques

- [ ] Implement cross-domain hybrid search (semantic + metadata across types)
- [ ] Theme extraction: identify common themes even when described differently
- [ ] Content balancing: ensure mix of movies/books/podcasts in results
- [ ] Format translation: 'books like the movie X' requires understanding narrative translation
- [ ] Temporal awareness: match pacing (quick podcast vs long book)
- [ ] Multi-hop reasoning: 'If they liked A, they might like B, so recommend C'
- [ ] Handle negation: 'Similar themes but NOT as technical'

### Performance Optimization

- [ ] Quantize models if needed (GGUF format for memory efficiency)
- [ ] Implement batch processing for embeddings
- [ ] Add caching strategies for frequent queries
- [ ] Monitor and optimize GPU memory usage
- [ ] Profile inference latency bottlenecks

---

## Phase 5: Production Polish
### Weeks 7-8

### Week 7: User Interface & Observability

**Tasks:**
- [ ] Build Gradio interface showing content cards across types
- [ ] Group recommendations by content type (Movies | Books | Podcasts)
- [ ] Show thematic connections: why these items match across formats
- [ ] Display agent reasoning: how theme analyzer identified patterns
- [ ] Add filtering: 'show only books' or 'prioritize podcasts'
- [ ] Track cross-domain query patterns and effectiveness
- [ ] Create dashboard: retrieval accuracy per content type, cross-domain success rate
- [ ] Implement feedback: 'this match was good/bad' per content type

**Example: Cross-Domain Content Interface**

```python
import gradio as gr

def discover_content(query, content_filter, history):
    # Get cross-domain recommendations
    result = crew.kickoff({
        "query": query,
        "content_types": content_filter or ["movie", "book", "podcast"]
    })
    
    # Group by content type
    movies = result.get("movies", [])
    books = result.get("books", [])
    podcasts = result.get("podcasts", [])
    
    # Format response
    response = "## Your Cross-Domain Recommendations\n\n"
    
    if movies:
        response += "### 🎬 Movies\n"
        for item in movies:
            response += f"**{item['title']}** - {item['explanation']}\n\n"
    
    if books:
        response += "### 📚 Books\n"
        for item in books:
            response += f"**{item['title']}** - {item['explanation']}\n\n"
    
    if podcasts:
        response += "### 🎙️ Podcasts\n"
        for item in podcasts:
            response += f"**{item['title']}** - {item['explanation']}\n\n"
    
    return response

demo = gr.ChatInterface(
    fn=discover_content,
    title="🌐 Cross-Domain Content Discovery",
    description="Find movies, books, and podcasts by theme, mood, or similarity"
)
demo.launch()
```

### Week 8: Documentation & Portfolio

**Tasks:**
- [ ] Write comprehensive README with architecture diagram
- [ ] Document setup and installation instructions
- [ ] Include performance benchmarks and comparisons
- [ ] Compare local GPU vs cloud API approaches
- [ ] Create requirements.txt and environment setup
- [ ] Add Docker support (optional but impressive)
- [ ] Record demo video showing end-to-end workflow
- [ ] Write blog post about technical challenges

---

## Success Metrics & Portfolio Highlights

### Key Metrics to Track

| Metric | What to Measure | Target |
|--------|-----------------|--------|
| Retrieval Accuracy | % relevant docs in top-k results | >80% |
| Answer Quality | Human evaluation rubric | >4/5 |
| Response Latency | End-to-end query time | <5 seconds |
| GPU Utilization | Memory usage during inference | 60-80% |
| Cross-Domain Success | Quality of matches across types | >70% |

### Portfolio Differentiators

- **Cross-Domain Reasoning**: Shows sophisticated semantic understanding across different content formats
- **Technical Ambition**: Multi-domain system is more impressive than single-domain
- **Real-World Applicability**: Same architecture powers e-commerce (products + reviews + videos)
- **Theme Extraction**: Demonstrates NLP depth beyond simple keyword matching
- **Production Thinking**: Document challenges in normalizing different data schemas
- **Scalability Discussion**: How to handle millions of items across content types
- **Transferability**: Clearly explain how this applies to enterprise knowledge bases with mixed document types

---

## Free Learning Resources (In Sequence)

| Week | Resource | Focus Area |
|------|----------|------------|
| 1-2 | Ollama Docs + LangChain RAG Tutorials | Setup & Basic RAG |
| 3 | LlamaIndex Documentation | Advanced Retrieval |
| 4 | CrewAI Docs + Examples | Agent Framework |
| 5 | DeepLearning.AI: AI Agents in LangGraph | Agent Orchestration |
| 5 | DeepLearning.AI: Agentic Design Patterns | Agent Patterns |
| 6-8 | Build and iterate based on findings | Custom Development |

---

## Additional Tips for Success

- **Start with Two Domains**: Begin with movies + books, add podcasts in Phase 4
- **Interview Framing**: Position as 'cross-domain semantic reasoning system' showcasing advanced RAG
- **Schema Design**: Invest time in unified content schema that works across all types
- **Test Cross-Domain Queries**: Focus on queries that span multiple content types
- **Document Challenges**: Write about normalizing different data formats (plot vs book summary vs episode description)
- **Version Control**: Git from day one with clear feature branches
- **Community Engagement**: Share cross-domain discovery insights on LinkedIn/Twitter
- **Blog Technical Wins**: Write about theme extraction across formats, schema design decisions
- **Transferability Focus**: Always connect to real-world use cases (enterprise docs, e-commerce, healthcare)

---

## Next Steps

Begin with Phase 1 immediately:
1. Install Ollama and pull your first model today
2. Sign up for free API keys (TMDb, Google Books, ListenNotes)
3. Start with a solid two-domain system (movies + books), then expand

The cross-domain content discovery system is more technically ambitious than single-domain projects and demonstrates sophisticated semantic reasoning that directly transfers to enterprise applications where you need to search across mixed document types (emails, PDFs, wikis, Slack messages).

**This architecture powers:**
- Multi-modal e-commerce search (products + reviews + videos)
- Enterprise knowledge bases (documents + code + conversations)
- Healthcare systems (research papers + clinical notes + patient records)
- Media platforms (articles + videos + podcasts)

**You're building production-relevant skills.**

---

## Real-World Applications

This same RAG architecture is used in:
- **Recommendation Systems**: Netflix, Spotify, Amazon
- **Customer Support**: Multi-source knowledge retrieval
- **Internal Knowledge Bases**: Search across Slack, Docs, Code, Email
- **Content Discovery**: Media companies finding related content across formats
- **E-commerce**: Product search across listings, reviews, videos, Q&A

---

Good luck with your Gen AI development journey! 🎬📚🎙️🚀
