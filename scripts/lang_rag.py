#!/usr/bin/env python3
# type: ignore
"""
RAG-based README generator and Q&A system for GEF Literature Repository
Uses LangChain with vector embeddings for intelligent document retrieval and processing
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pickle

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks import StreamingStdOutCallbackHandler

# Additional imports
import openai
from openai import OpenAI

class GEFDocumentProcessor:
    """Handles document loading, processing, and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitters
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def find_markdown_files(self, directory: Path) -> List[Path]:
        """Find all markdown files excluding README.md"""
        markdown_files = []
        for item in directory.rglob("*.md"):
            if (not any(part.startswith('.') for part in item.parts) and 
                item.name != "README.md"):
                markdown_files.append(item)
        return sorted(markdown_files)
    
    def load_documents(self, directory: Path) -> List[Document]:
        """Load and process all markdown files into LangChain Documents"""
        markdown_files = self.find_markdown_files(directory)
        documents = []
        
        print(f"📄 Loading {len(markdown_files)} markdown files...")
        
        for file_path in markdown_files:
            try:
                relative_path = file_path.relative_to(directory)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create metadata
                metadata = {
                    'source': str(relative_path),
                    'file_name': file_path.name,
                    'folder': str(relative_path.parent),
                    'file_size': len(content),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # Create Document object
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load {file_path}: {e}")
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using hierarchical splitting"""
        all_chunks = []
        
        print("🔪 Chunking documents...")
        
        for doc in documents:
            try:
                # First split by markdown headers
                header_chunks = self.markdown_splitter.split_text(doc.page_content)
                
                # Then split large chunks further
                final_chunks = []
                for chunk in header_chunks:
                    if isinstance(chunk, Document):
                        chunk_content = chunk.page_content
                        chunk_metadata = chunk.metadata
                    else:
                        chunk_content = chunk
                        chunk_metadata = {}
                    
                    if len(chunk_content) > self.chunk_size:
                        sub_chunks = self.text_splitter.split_text(chunk_content)
                        for sub_chunk in sub_chunks:
                            # Combine metadata
                            combined_metadata = {**doc.metadata, **chunk_metadata}
                            combined_metadata['chunk_type'] = 'sub_chunk'
                            final_chunks.append(Document(
                                page_content=sub_chunk,
                                metadata=combined_metadata
                            ))
                    else:
                        combined_metadata = {**doc.metadata, **chunk_metadata}
                        combined_metadata['chunk_type'] = 'header_chunk'
                        final_chunks.append(Document(
                            page_content=chunk_content,
                            metadata=combined_metadata
                        ))
                
                all_chunks.extend(final_chunks)
                
            except Exception as e:
                print(f"⚠️  Warning: Could not chunk document {doc.metadata.get('source', 'unknown')}: {e}")
                # Fallback: use simple text splitting
                simple_chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(simple_chunks)
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

class GEFVectorStore:
    """Handles vector store creation and management"""
    
    def __init__(self, embeddings_model: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vector_store = None
        self.store_path = Path("./vector_store")
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from document chunks"""
        print("🔍 Creating vector embeddings...")
        
        try:
            # Create vector store
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            print(f"✅ Created vector store with {len(chunks)} embeddings")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def save_vector_store(self) -> None:
        """Save vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        self.store_path.mkdir(exist_ok=True)
        self.vector_store.save_local(str(self.store_path))
        print(f"💾 Vector store saved to {self.store_path}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load vector store from disk"""
        if not self.store_path.exists():
            return None
        
        try:
            self.vector_store = FAISS.load_local(str(self.store_path), self.embeddings)
            print(f"📖 Vector store loaded from {self.store_path}")
            return self.vector_store
        except Exception as e:
            print(f"⚠️  Could not load vector store: {e}")
            return None

class GEFRAGSystem:
    """Main RAG system for summary generation and Q&A"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.summary_chain = None
        
        # Setup prompts
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup custom prompts for different use cases"""
        
        # Q&A Prompt
        self.qa_prompt = PromptTemplate(
            template="""You are an expert research assistant specializing in Global Energy Forecasting (GEF) literature analysis.

Use the following pieces of context from the research repository to answer the question. If you don't know the answer based on the context, say that you don't have enough information in the repository to answer that question.

Context from research documents:
{context}

Question: {question}

Provide a detailed, well-structured answer based on the research documents. Include:
1. Direct information from the documents
2. Relevant methodologies or findings mentioned
3. References to specific files/folders when relevant
4. Any limitations in the available information

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Summary Generation Prompt
        self.summary_prompt = PromptTemplate(
            template="""You are analyzing research documents from a Global Energy Forecasting (GEF) literature repository.

Based on the following representative content from the repository, create a comprehensive summary:

Repository Content:
{context}

Create a detailed summary with the following structure:

## 🎯 Research Overview
Describe the main research focus, objectives, and scope of the GEF literature repository.

## 🔬 Key Research Areas
Identify and explain the main research topics, methodologies, and approaches found in the documents.

## 📁 Repository Organization
Explain how the research is organized (folder structure, naming conventions, progression over time).

## 💡 Key Findings & Insights
Highlight the most important discoveries, conclusions, and insights from the research.

## 🛠️ Methodologies & Tools
List and describe the main research methodologies, datasets, software tools, and frameworks mentioned.

## 📚 Literature & References
Identify key papers, authors, and external resources referenced in the research.

## 📈 Research Evolution
Describe how the research has progressed over time based on the folder/file organization.

Focus on the most significant and valuable content. Be specific and include examples where possible.

Summary:""",
            input_variables=["context"]
        )
    
    def setup_qa_chain(self, vector_store: FAISS) -> RetrievalQA:
        """Setup the Q&A chain with retrieval"""
        self.vector_store = vector_store
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Retrieve top 8 most relevant chunks
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def generate_summary(self, vector_store: FAISS, max_chunks: int = 20) -> str:
        """Generate comprehensive summary using RAG"""
        print("🤖 Generating comprehensive summary using RAG...")
        
        # Get diverse representative chunks for summary
        # Use different search queries to get varied content
        search_queries = [
            "research methodology global energy forecasting",
            "key findings results energy prediction",
            "literature review energy forecasting models",
            "dataset analysis energy consumption",
            "forecasting accuracy evaluation metrics",
            "time series analysis energy data"
        ]
        
        representative_chunks = []
        seen_sources = set()
        
        for query in search_queries:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max_chunks // len(search_queries)}
            )
            
            docs = retriever.get_relevant_documents(query)
            
            for doc in docs:
                source = doc.metadata.get('source', '')
                # Avoid duplicates and ensure diversity
                if source not in seen_sources and len(representative_chunks) < max_chunks:
                    representative_chunks.append(doc)
                    seen_sources.add(source)
        
        # If we don't have enough diverse content, add more chunks
        if len(representative_chunks) < max_chunks:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max_chunks - len(representative_chunks)}
            )
            additional_docs = retriever.get_relevant_documents("global energy forecasting research")
            
            for doc in additional_docs:
                source = doc.metadata.get('source', '')
                if source not in seen_sources and len(representative_chunks) < max_chunks:
                    representative_chunks.append(doc)
                    seen_sources.add(source)
        
        # Combine chunk content
        context = "\n\n---\n\n".join([
            f"**File: {chunk.metadata.get('source', 'unknown')}**\n{chunk.page_content}"
            for chunk in representative_chunks
        ])
        
        # Generate summary
        try:
            prompt = self.summary_prompt.format(context=context)
            response = self.llm.predict(prompt)
            return response
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Error generating summary. Repository contains research on Global Energy Forecasting with {len(seen_sources)} files analyzed."
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question using the RAG system"""
        if self.qa_chain is None:
            raise ValueError("Q&A chain not initialized. Call setup_qa_chain first.")
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "folder": doc.metadata.get("folder", "unknown")
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {e}",
                "source_documents": []
            }

def create_readme_content(summary: str, file_count: int, chunk_count: int) -> str:
    """Create the README.md content"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""# Global Energy Forecasting (GEF) Literature Repository

*Last updated: {timestamp}*
*Files analyzed: {file_count} | Chunks processed: {chunk_count}*

## 🤖 RAG-Generated Comprehensive Summary

{summary}

---

## 🔍 Interactive Q&A System

This repository includes an intelligent Q&A system powered by RAG (Retrieval-Augmented Generation). You can ask specific questions about the research content:

```bash
# Ask questions about the research
python rag_system.py --mode qa --question "What are the main forecasting methodologies used?"
python rag_system.py --mode qa --question "Which datasets were analyzed in the research?"
python rag_system.py --mode qa --question "What are the key findings about energy consumption patterns?"
```

## 📁 Repository Structure

This repository contains research notes and documentation organized by date in `RUN_XX_X` folders, with intelligent vector-based search and retrieval capabilities.

## 🚀 Features

- **RAG-based Analysis**: Uses vector embeddings for intelligent document retrieval
- **Comprehensive Summaries**: AI-generated overviews of all research content
- **Interactive Q&A**: Ask specific questions about your research
- **Smart Chunking**: Hierarchical document splitting for better context preservation
- **Vector Search**: Find relevant content across all documents instantly

## 🛠️ Usage

### Generate Summary
```bash
python rag_system.py --mode summary
```

### Ask Questions
```bash
python rag_system.py --mode qa --question "Your question here"
```

### Rebuild Vector Store
```bash
python rag_system.py --mode rebuild
```

## 📝 Contributing

When adding new research notes:
1. Create markdown files in the appropriate `RUN_XX_X` folder
2. Use clear headers and structured content
3. Run the rebuild command to update the vector store
4. Generate a new summary to reflect changes

## 🔧 Technical Details

- **Vector Store**: FAISS with OpenAI embeddings
- **LLM**: GPT-3.5-turbo-16k for generation
- **Framework**: LangChain for RAG implementation
- **Chunking**: Hierarchical splitting preserving document structure
- **Search**: Semantic similarity search across all content

---
*This repository uses advanced RAG technology to make your research searchable and queryable. Ask questions, generate summaries, and explore your research in new ways!*"""

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="GEF RAG System")
    parser.add_argument("--mode", choices=["summary", "qa", "rebuild"], required=True,
                       help="Operation mode: generate summary, Q&A, or rebuild vector store")
    parser.add_argument("--question", type=str, help="Question for Q&A mode")
    parser.add_argument("--directory", type=str, default=".", help="Repository directory")
    
    args = parser.parse_args()
    
    # Verify OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        repo_root = Path(args.directory)
        
        # Initialize components
        doc_processor = GEFDocumentProcessor()
        vector_store_manager = GEFVectorStore()
        rag_system = GEFRAGSystem()
        
        if args.mode == "rebuild" or not vector_store_manager.store_path.exists():
            print("🔄 Building vector store...")
            
            # Load and process documents
            documents = doc_processor.load_documents(repo_root)
            if not documents:
                print("❌ No markdown files found")
                sys.exit(1)
            
            chunks = doc_processor.chunk_documents(documents)
            vector_store = vector_store_manager.create_vector_store(chunks)
            vector_store_manager.save_vector_store()
            
            if args.mode == "rebuild":
                print("✅ Vector store rebuilt successfully!")
                return
        else:
            # Load existing vector store
            vector_store = vector_store_manager.load_vector_store()
            if vector_store is None:
                print("❌ Could not load vector store. Run with --mode rebuild first.")
                sys.exit(1)
        
        if args.mode == "summary":
            print("📝 Generating comprehensive summary...")
            
            # Count files for README
            files = doc_processor.find_markdown_files(repo_root)
            file_count = len(files)
            
            # Get chunk count from vector store
            chunk_count = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0
            
            # Generate summary
            summary = rag_system.generate_summary(vector_store)
            
            # Create README
            readme_content = create_readme_content(summary, file_count, chunk_count)
            
            with open(repo_root / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print("✅ README.md updated successfully!")
            
        elif args.mode == "qa":
            if not args.question:
                print("❌ Question required for Q&A mode. Use --question 'Your question here'")
                sys.exit(1)
            
            print(f"❓ Processing question: {args.question}")
            
            # Setup Q&A chain
            rag_system.setup_qa_chain(vector_store)
            
            # Ask question
            result = rag_system.ask_question(args.question)
            
            print("\n" + "="*80)
            print("🤖 ANSWER:")
            print("="*80)
            print(result["answer"])
            
            if result["source_documents"]:
                print("\n" + "="*80)
                print("📚 SOURCES:")
                print("="*80)
                for i, source in enumerate(result["source_documents"], 1):
                    print(f"\n{i}. **{source['source']}** (folder: {source['folder']})")
                    print(f"   {source['content']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()