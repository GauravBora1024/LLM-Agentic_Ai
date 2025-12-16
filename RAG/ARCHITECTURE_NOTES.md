# RAG System Architecture and Evaluation Framework
## Academic Documentation

### Abstract
This document provides a comprehensive analysis of a Retrieval-Augmented Generation (RAG) system implementation with integrated evaluation framework. The system is designed for question-answering over a knowledge base about Insurellm, an insurance technology company. The architecture demonstrates a modular design separating implementation concerns (data ingestion, retrieval, generation) from evaluation concerns (retrieval metrics, answer quality assessment).

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture
The system follows a three-tier architecture:

1. **Data Layer**: Document ingestion and vector database management
2. **Application Layer**: RAG pipeline for question answering
3. **Evaluation Layer**: Comprehensive evaluation framework with multiple metrics

### 1.2 Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Interface                     │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   app.py     │              │ evaluator.py │            │
│  │ (Chat UI)    │              │ (Eval UI)    │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼─────────────────────────────┼────────────────────┘
          │                             │
          │                             │
┌─────────┼─────────────────────────────┼────────────────────┐
│         │                             │                    │
│  ┌──────▼───────┐            ┌───────▼────────┐          │
│  │ answer.py    │            │  eval.py        │          │
│  │ (RAG Core)   │◄───────────┤ (Evaluation)   │          │
│  └──────┬───────┘            └───────┬─────────┘          │
│         │                            │                    │
│  ┌──────▼───────┐            ┌───────▼────────┐          │
│  │ ingest.py    │            │  test.py       │          │
│  │ (Ingestion)  │            │ (Test Loader)  │          │
│  └──────┬───────┘            └───────┬────────┘          │
│         │                            │                    │
└─────────┼─────────────────────────────┼────────────────────┘
          │                             │
          ▼                             ▼
    ┌─────────────┐              ┌──────────────┐
    │ Vector DB   │              │ tests.jsonl  │
    │ (Chroma)    │              │ (Test Data)  │
    └─────────────┘              └──────────────┘
```

---

## 2. Implementation Components

### 2.1 Data Ingestion Module (`implementation/ingest.py`)

**Purpose**: Transforms raw documents into a searchable vector database.

**Key Functions**:
- `fetch_documents()`: Recursively loads markdown files from knowledge base directory structure
- `create_chunks()`: Splits documents using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
- `create_embeddings()`: Generates embeddings using OpenAI's `text-embedding-3-large` model and persists to Chroma vector database

**Technical Details**:
- **Embedding Model**: OpenAI `text-embedding-3-large` (3072 dimensions)
- **Vector Store**: Chroma (persistent storage at `vector_db_v2/`)
- **Chunking Strategy**: Recursive character splitting with 200-token overlap to preserve context boundaries
- **Metadata**: Preserves document type information from folder structure

**Data Flow**:
```
Knowledge Base (Markdown Files)
    ↓
DirectoryLoader (recursive)
    ↓
Text Splitter (chunking)
    ↓
Embedding Generation
    ↓
Chroma Vector Store (persistent)
```

### 2.2 RAG Core Module (`implementation/answer.py`)

**Purpose**: Implements the retrieval-augmented generation pipeline for answering questions.

**Key Functions**:
- `fetch_context(question: str)`: Retrieves top-k (k=5) relevant documents using semantic search
- `combined_question(question: str, history: list)`: Aggregates conversation history for context-aware retrieval
- `answer_question(question: str, history: list)`: Complete RAG pipeline returning answer and retrieved documents

**Technical Details**:
- **Retrieval**: Chroma retriever with k=5 documents
- **LLM**: GPT-4.1-nano (temperature=0 for deterministic responses)
- **Retrieval Strategy**: Semantic similarity search using cosine distance in embedding space
- **Context Assembly**: Concatenates retrieved documents with newline separators
- **Prompt Engineering**: System prompt instructs model to use context and admit uncertainty

**RAG Pipeline**:
```
User Question + History
    ↓
Semantic Retrieval (k=5)
    ↓
Context Assembly
    ↓
LLM Generation (with context)
    ↓
Answer + Retrieved Documents
```

**System Prompt Structure**:
```
You are a knowledgeable, friendly assistant representing Insurellm.
Use the given context to answer questions.
If you don't know the answer, say so.
Context: {retrieved_documents}
```

---

## 3. Evaluation Framework

### 3.1 Test Data Structure (`evaluation/test.py` & `tests.jsonl`)

**Purpose**: Defines schema and loads evaluation test cases.

**TestQuestion Schema** (Pydantic BaseModel):
- `question`: Natural language question to evaluate
- `keywords`: List of expected keywords that should appear in retrieved context
- `reference_answer`: Ground truth answer for comparison
- `category`: Question type classification (direct_fact, temporal, comparative, numerical, relationship, spanning, holistic)

**Test Dataset**:
- **Format**: JSONL (151 test cases)
- **Categories**: 8 distinct question types
- **Coverage**: Comprehensive evaluation across different query complexities

**Category Distribution**:
- `direct_fact`: Simple factual queries
- `temporal`: Time-based queries
- `comparative`: Comparison queries
- `numerical`: Quantitative queries
- `relationship`: Entity relationship queries
- `spanning`: Multi-hop reasoning queries
- `holistic`: Aggregation queries requiring multiple facts

### 3.2 Retrieval Evaluation (`evaluation/eval.py` - Retrieval Metrics)

**Purpose**: Evaluates the quality of document retrieval independent of answer generation.

**Metrics Implemented**:

1. **Mean Reciprocal Rank (MRR)**
   - Formula: `MRR = (1/n) * Σ(1/rank_i)` where rank_i is the position of the first relevant document for keyword i
   - Interpretation: Higher values (0-1) indicate better retrieval performance
   - Calculation: Case-insensitive keyword matching in retrieved documents

2. **Normalized Discounted Cumulative Gain (nDCG)**
   - Formula: `nDCG = DCG / IDCG`
   - DCG: `Σ(relevance_i / log2(i+2))` for top-k results
   - Binary relevance: 1 if keyword found, 0 otherwise
   - Normalization: Divides by ideal DCG (perfect ranking scenario)
   - Interpretation: Measures ranking quality with position discounting

3. **Keyword Coverage**
   - Formula: `(keywords_found / total_keywords) * 100`
   - Interpretation: Percentage of expected keywords present in retrieved documents

**RetrievalEval Schema**:
```python
class RetrievalEval(BaseModel):
    mrr: float                    # Mean Reciprocal Rank
    ndcg: float                   # Normalized DCG
    keywords_found: int           # Count of keywords found
    total_keywords: int           # Total keywords expected
    keyword_coverage: float       # Percentage coverage
```

**Evaluation Process**:
```
Test Question
    ↓
fetch_context() [from answer.py]
    ↓
Calculate MRR for each keyword
    ↓
Calculate nDCG for each keyword
    ↓
Aggregate metrics (average across keywords)
    ↓
RetrievalEval result
```

### 3.3 Answer Quality Evaluation (`evaluation/eval.py` - LLM-as-a-Judge)

**Purpose**: Evaluates generated answers using an LLM judge with structured output.

**Evaluation Dimensions**:

1. **Accuracy** (1-5 scale)
   - Measures factual correctness compared to reference answer
   - Score 1: Wrong answer (any incorrect information)
   - Score 3: Acceptable accuracy
   - Score 5: Perfect accuracy (all facts correct)

2. **Completeness** (1-5 scale)
   - Measures thoroughness in addressing all question aspects
   - Score 1: Missing key information
   - Score 5: All information from reference answer included

3. **Relevance** (1-5 scale)
   - Measures directness and focus of the answer
   - Score 1: Off-topic or irrelevant
   - Score 5: Directly addresses question with no extraneous information

**AnswerEval Schema**:
```python
class AnswerEval(BaseModel):
    feedback: str                 # Qualitative feedback
    accuracy: float               # 1-5 scale
    completeness: float           # 1-5 scale
    relevance: float              # 1-5 scale
```

**LLM Judge Prompt Structure**:
```
System: Expert evaluator assessing answer quality
User: 
  Question: {test.question}
  Generated Answer: {generated_answer}
  Reference Answer: {test.reference_answer}
  
  Evaluate on:
  1. Accuracy (factual correctness)
  2. Completeness (thoroughness)
  3. Relevance (directness)
```

**Evaluation Process**:
```
Test Question
    ↓
answer_question() [from answer.py]
    ↓
LLM Judge (structured output)
    ↓
AnswerEval result
```

**Methodology**: Uses "LLM-as-a-Judge" paradigm where GPT-4.1-nano evaluates answers with structured JSON output via Pydantic validation.

### 3.4 Evaluation Orchestration

**Batch Evaluation Functions**:
- `evaluate_all_retrieval()`: Generator yielding (test, RetrievalEval, progress) tuples
- `evaluate_all_answers()`: Generator yielding (test, AnswerEval, progress) tuples

**CLI Evaluation**:
- `run_cli_evaluation(test_number: int)`: Runs both retrieval and answer evaluation for a single test case
- Provides detailed console output with all metrics

---

## 4. User Interfaces

### 4.1 Chat Interface (`app.py`)

**Purpose**: Interactive Gradio-based chat interface for end-users.

**Features**:
- Real-time question answering
- Conversation history support
- Context visualization (shows retrieved documents)
- Source attribution for retrieved documents

**UI Components**:
- Chatbot interface (message-based)
- Context display panel (markdown formatted)
- Source metadata display

**User Flow**:
```
User Input
    ↓
answer_question() with history
    ↓
Display Answer + Retrieved Context
```

### 4.2 Evaluation Dashboard (`evaluator.py`)

**Purpose**: Comprehensive evaluation interface with visualization.

**Features**:
- Separate evaluation modes for retrieval and answer quality
- Real-time progress tracking
- Color-coded metrics (green/amber/red thresholds)
- Category-based performance breakdown
- Bar charts for category analysis

**Color Coding Thresholds**:

**Retrieval Metrics**:
- MRR: Green ≥0.9, Amber ≥0.75, Red <0.75
- nDCG: Green ≥0.9, Amber ≥0.75, Red <0.75
- Coverage: Green ≥90%, Amber ≥75%, Red <75%

**Answer Metrics** (1-5 scale):
- Green: ≥4.5
- Amber: ≥4.0
- Red: <4.0

**Dashboard Sections**:
1. **Retrieval Evaluation**: MRR, nDCG, Keyword Coverage with category breakdown
2. **Answer Evaluation**: Accuracy, Completeness, Relevance with category breakdown

**Visualization**:
- HTML-formatted metric cards with color coding
- Bar plots showing average performance by question category
- Progress indicators during evaluation

---

## 5. System Dependencies and Integration

### 5.1 Shared Components

**Vector Database**:
- Both `answer.py` and `eval.py` access the same Chroma vector store
- Ensures evaluation uses identical retrieval as production system
- Path: `vector_db_v2/` (relative to week5 directory)

**LLM Configuration**:
- Model: `gpt-4.1-nano` (consistent across all modules)
- Temperature: 0 (deterministic responses)
- API: LiteLLM for completion calls

**Embedding Model**:
- OpenAI `text-embedding-3-large` (3072 dimensions)
- Used for both ingestion and retrieval

### 5.2 Module Dependencies

```
app.py
  └─> implementation.answer
      └─> Chroma vector store

evaluator.py
  └─> evaluation.eval
      ├─> evaluation.test (loads tests.jsonl)
      └─> implementation.answer (for RAG pipeline)

eval.py
  ├─> evaluation.test
  └─> implementation.answer

test.py
  └─> tests.jsonl (data file)

answer.py
  ├─> Chroma vector store
  └─> OpenAI LLM

ingest.py
  ├─> Knowledge base (markdown files)
  └─> Chroma vector store (creates/updates)
```

---

## 6. Evaluation Methodology

### 6.1 Two-Stage Evaluation Approach

The system employs a two-stage evaluation strategy:

**Stage 1: Retrieval Evaluation**
- Evaluates the information retrieval component in isolation
- Metrics: MRR, nDCG, Keyword Coverage
- Purpose: Assess whether relevant information is being retrieved

**Stage 2: Answer Quality Evaluation**
- Evaluates the complete RAG pipeline (retrieval + generation)
- Metrics: Accuracy, Completeness, Relevance (LLM-as-a-Judge)
- Purpose: Assess whether retrieved information is correctly synthesized into answers

### 6.2 Evaluation Design Principles

1. **Separation of Concerns**: Retrieval and generation evaluated independently
2. **Ground Truth Comparison**: Reference answers provide objective benchmarks
3. **Multi-Dimensional Assessment**: Multiple metrics capture different quality aspects
4. **Category-Based Analysis**: Performance breakdown by question type
5. **Automated Evaluation**: LLM-as-a-Judge enables scalable assessment

### 6.3 Keyword-Based Retrieval Evaluation

**Rationale**: Keywords serve as proxies for document relevance. If expected keywords appear in retrieved documents, the retrieval is likely successful.

**Limitations**:
- Binary relevance (keyword present/absent)
- Does not capture semantic relevance without explicit keywords
- May miss relevant documents using different terminology

**Strengths**:
- Objective and reproducible
- Fast computation
- Clear interpretability

### 6.4 LLM-as-a-Judge Evaluation

**Rationale**: LLMs can assess answer quality by comparing generated answers to reference answers across multiple dimensions.

**Advantages**:
- Captures semantic similarity beyond exact matching
- Multi-dimensional assessment (accuracy, completeness, relevance)
- Scalable to large test sets

**Considerations**:
- Subject to LLM biases and inconsistencies
- Requires careful prompt engineering
- Structured output ensures consistency

---

## 7. Data Flow Analysis

### 7.1 Ingestion Pipeline

```
Knowledge Base (Markdown Files)
    ↓ [DirectoryLoader]
Raw Documents
    ↓ [RecursiveCharacterTextSplitter]
Document Chunks (1000 tokens, 200 overlap)
    ↓ [OpenAI Embeddings]
Vector Embeddings (3072 dimensions)
    ↓ [Chroma.from_documents]
Persistent Vector Database
```

### 7.2 Query Pipeline

```
User Question
    ↓ [Embedding]
Query Vector
    ↓ [Semantic Search - Chroma]
Top-k Documents (k=5)
    ↓ [Context Assembly]
Formatted Context String
    ↓ [LLM Generation]
Final Answer
```

### 7.3 Evaluation Pipeline

```
Test Question (from tests.jsonl)
    ↓
┌─────────────────┬──────────────────┐
│ Retrieval Eval  │  Answer Eval     │
│                 │                   │
│ fetch_context() │ answer_question()│
│ ↓               │ ↓                 │
│ MRR/nDCG        │ LLM Judge         │
│ Calculation     │ Evaluation        │
└─────────────────┴──────────────────┘
    ↓
Aggregated Metrics
    ↓
Visualization (Dashboard)
```

---

## 8. Technical Specifications

### 8.1 Vector Database
- **Technology**: Chroma
- **Storage**: Persistent on-disk storage
- **Embedding Dimensions**: 3072 (text-embedding-3-large)
- **Retrieval Method**: Cosine similarity search
- **Top-k**: 5 documents per query

### 8.2 Language Models
- **Embedding Model**: OpenAI `text-embedding-3-large`
- **Generation Model**: GPT-4.1-nano (via LiteLLM)
- **Judge Model**: GPT-4.1-nano (via LiteLLM with structured output)

### 8.3 Text Processing
- **Chunking**: RecursiveCharacterTextSplitter
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Document Format**: Markdown (.md files)

### 8.4 Evaluation Metrics
- **Retrieval**: MRR, nDCG (k=10), Keyword Coverage
- **Answer Quality**: Accuracy, Completeness, Relevance (1-5 scale)
- **Aggregation**: Mean across all test cases, mean by category

---

## 9. Research Contributions and Design Patterns

### 9.1 Modular Architecture
The system demonstrates clean separation between:
- **Data ingestion** (ingest.py)
- **Core RAG functionality** (answer.py)
- **Evaluation framework** (eval.py, test.py)
- **User interfaces** (app.py, evaluator.py)

This modularity enables:
- Independent testing of components
- Easy replacement of components (e.g., different embedding models)
- Clear separation of concerns

### 9.2 Evaluation Best Practices
1. **Multi-metric Assessment**: Uses both retrieval and generation metrics
2. **Category-Based Analysis**: Breaks down performance by question type
3. **Automated Evaluation**: LLM-as-a-Judge for scalable assessment
4. **Ground Truth Comparison**: Reference answers provide objective benchmarks

### 9.3 RAG System Design
1. **Context-Aware Retrieval**: Incorporates conversation history
2. **Source Attribution**: Tracks document sources for transparency
3. **Uncertainty Handling**: System prompt instructs model to admit uncertainty
4. **Deterministic Responses**: Temperature=0 for reproducible outputs

---

## 10. Limitations and Future Work

### 10.1 Current Limitations
1. **Keyword-Based Evaluation**: Binary relevance may miss semantic matches
2. **Fixed Retrieval k**: Always retrieves 5 documents regardless of query complexity
3. **No Reranking**: Single-stage retrieval without reranking
4. **LLM Judge Consistency**: Subject to model variability

### 10.2 Potential Enhancements
1. **Hybrid Retrieval**: Combine semantic and keyword-based search
2. **Adaptive k**: Retrieve different numbers of documents based on query
3. **Reranking**: Add cross-encoder reranking stage
4. **Human Evaluation**: Supplement LLM judge with human annotations
5. **A/B Testing**: Compare different retrieval strategies

---

## 11. Conclusion

This RAG system demonstrates a production-ready architecture with comprehensive evaluation capabilities. The modular design facilitates maintenance and extension, while the evaluation framework provides actionable insights into system performance. The integration of retrieval and generation evaluation enables holistic assessment of RAG system quality.

The system successfully implements:
- Efficient document ingestion and vector storage
- Semantic retrieval with context awareness
- LLM-based answer generation
- Multi-dimensional evaluation framework
- User-friendly interfaces for both end-users and evaluators

This architecture serves as a solid foundation for question-answering systems over domain-specific knowledge bases.

---

## References

### Key Technologies
- LangChain: Framework for LLM applications
- Chroma: Vector database for embeddings
- OpenAI: Embedding and generation models
- LiteLLM: Unified LLM API interface
- Gradio: Web interface framework
- Pydantic: Data validation and schema definition

### Evaluation Metrics
- Mean Reciprocal Rank (MRR): Information retrieval metric
- Normalized Discounted Cumulative Gain (nDCG): Ranking quality metric
- LLM-as-a-Judge: Evaluation paradigm using LLMs as evaluators

---

*Document generated: Academic analysis of RAG system architecture*
*Last updated: Based on codebase analysis*

