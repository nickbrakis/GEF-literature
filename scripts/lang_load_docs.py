# ...existing code...
#!/usr/bin/env python3
# type: ignore
"""
LangChain Tutorial: Understanding Documents and Text Processing
Step 1: Basic concepts with simple examples
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def load_workspace_markdown_files():
    """
    Load all markdown files from the GEF literature workspace as LangChain Documents
    Excludes README.md files to focus on research content
    """
    
    print("📚 Loading Workspace Markdown Files as Documents")
    print("=" * 50)
    
    from pathlib import Path
    
    # Get the workspace root directory (go up one level from scripts)
    workspace_root = Path(__file__).parent.parent
    print(f"Workspace root: {workspace_root}")
    
    # Find all markdown files, excluding README.md
    markdown_files = []
    for md_file in workspace_root.rglob("*.md"):
        # Skip hidden directories and README files
        if (not any(part.startswith('.') for part in md_file.parts) and 
            md_file.name != "README.md"):
            markdown_files.append(md_file)
    
    print(f"Found {len(markdown_files)} markdown files")
    
    # Create Documents from each file
    documents = []
    for file_path in sorted(markdown_files):
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract folder/run information
            relative_path = file_path.relative_to(workspace_root)
            folder_name = relative_path.parent.name
            
            # Create metadata with useful information
            metadata = {
                "source": str(relative_path),
                "filename": file_path.name,
                "folder": folder_name,
                "full_path": str(file_path),
                "size": len(content),
                "file_type": "research_notes"
            }
            
            # Add run-specific metadata if it's a RUN folder
            if folder_name.startswith("RUN_"):
                try:
                    # Extract date from folder name (e.g., RUN_6_27 -> month 6, day 27)
                    parts = folder_name.split("_")
                    if len(parts) >= 3:
                        metadata["run_month"] = parts[1]
                        metadata["run_day"] = parts[2]
                        metadata["run_id"] = folder_name
                except:
                    pass
            
            # Determine content type based on filename
            filename_lower = file_path.name.lower()
            if "clustering" in filename_lower:
                metadata["topic"] = "clustering"
            elif "darts" in filename_lower:
                metadata["topic"] = "darts_framework"
            elif "energy" in filename_lower or "load" in filename_lower or "forecasting" in filename_lower:
                metadata["topic"] = "energy_forecasting"
            elif "eda" in filename_lower:
                metadata["topic"] = "exploratory_data_analysis"
            elif "dataset" in filename_lower:
                metadata["topic"] = "dataset_analysis"
            elif "pipeline" in filename_lower:
                metadata["topic"] = "data_pipeline"
            elif "paper" in filename_lower or "review" in filename_lower or "stlf" in filename_lower:
                metadata["topic"] = "literature_review"
            elif "deeptsf" in filename_lower:
                metadata["topic"] = "deeptsf_platform"
            else:
                metadata["topic"] = "general"
            
            # Create Document
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
            print(f"✅ Loaded: {relative_path} ({len(content):,} chars, topic: {metadata['topic']})")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"Total documents: {len(documents)}")
    print(f"Total content: {sum(len(doc.page_content) for doc in documents):,} characters")
    
    # Show topic distribution
    topics = {}
    for doc in documents:
        topic = doc.metadata.get("topic", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
    
    print(f"\n📋 Content by topic:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count} files")
    
    return documents

def create_smart_chunks_from_workspace(documents):
    """
    Load workspace files and create optimally-sized chunks for RAG
    """
    
    print("\n🧩 Creating Smart Chunks from Workspace")
    print("=" * 50)
    
    
    # Configure splitters
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
        ]
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Larger chunks for better context
        chunk_overlap=200,  # Good overlap for continuity
        length_function=len
    )
    
    all_chunks = []
    
    for doc in documents:
        print(f"\nProcessing: {doc.metadata['source']}")
        
        # Step 1: Try markdown header splitting first
        try:
            header_chunks = markdown_splitter.split_text(doc.page_content)
            
            # Step 2: Further split large chunks
            for chunk in header_chunks:
                if isinstance(chunk, Document):
                    content = chunk.page_content
                    chunk_metadata = {**doc.metadata, **chunk.metadata}
                else:
                    content = chunk
                    chunk_metadata = doc.metadata.copy()
                
                # If chunk is still too large, split further
                if len(content) > 1200:  # Slightly larger than chunk_size for flexibility
                    sub_chunks = text_splitter.split_text(content)
                    for i, sub_chunk in enumerate(sub_chunks):
                        final_metadata = chunk_metadata.copy()
                        final_metadata["chunk_index"] = i
                        final_metadata["chunk_method"] = "header_then_recursive"
                        
                        all_chunks.append(Document(
                            page_content=sub_chunk,
                            metadata=final_metadata
                        ))
                else:
                    chunk_metadata["chunk_method"] = "header_only"
                    all_chunks.append(Document(
                        page_content=content,
                        metadata=chunk_metadata
                    ))
        
        except Exception as e:
            print(f"  ⚠️ Header splitting failed, using recursive splitter: {e}")
            # Fallback to recursive splitting
            chunks = text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_method"] = "recursive_only"
                all_chunks.append(chunk)
    
    print(f"\n📊 Chunking Results:")
    print(f"Original documents: {len(documents)}")
    print(f"Final chunks: {len(all_chunks)}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} chars")
    
    # Show chunk size distribution
    sizes = [len(chunk.page_content) for chunk in all_chunks]
    print(f"Chunk sizes - Min: {min(sizes)}, Max: {max(sizes)}, Median: {sorted(sizes)[len(sizes)//2]}")
    
    return all_chunks

def filter_documents_by_topic(documents, topics):
    """
    Filter documents by specific topics
    
    Args:
        documents: List of Document objects
        topics: List of topic strings to include
    
    Returns:
        Filtered list of documents
    """
    filtered = [doc for doc in documents if doc.metadata.get("topic") in topics]
    
    print(f"Filtered {len(documents)} documents to {len(filtered)} matching topics: {topics}")
    return filtered

def demo_workspace_loading():
    """
    Demonstrate loading and processing workspace markdown files
    """
    
    print("🎯 Demo: Loading GEF Literature Workspace")
    print("=" * 60)
    
    # Load all documents
    documents = load_workspace_markdown_files()
    
    input("\nPress Enter to create smart chunks...")
    
    # Create optimized chunks
    chunks = create_smart_chunks_from_workspace(documents)
    
    input("\nPress Enter to see example filtering...")
    
    # Example: Filter for only clustering and forecasting papers
    research_docs = filter_documents_by_topic(documents, ["clustering", "energy_forecasting", "literature_review"])
    
    print(f"\n📋 Example Documents:")
    for i, doc in enumerate(research_docs[:3]):  # Show first 3
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Topic: {doc.metadata['topic']}")
        print(f"Preview: {doc.page_content[:200]}...")
    
    return documents, chunks

# Add this to the main function
if __name__ == "__main__":
    # Uncomment to run the workspace demo instead of the tutorial
    demo_workspace_loading()
    
    # Or run the original tutorial
    # main()