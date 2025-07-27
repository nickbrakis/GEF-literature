#!/usr/bin/env python3
# type: ignore
"""
LangChain Tutorial: Understanding Documents and Text Processing
Step 1: Basic concepts with simple examples
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# =============================================================================
# 1. DOCUMENTS - The Basic Building Block
# =============================================================================

def understand_documents():
    """
    In LangChain, everything starts with Documents.
    A Document is just text + metadata (information about the text)
    """
    
    print("📄 Understanding LangChain Documents")
    print("=" * 50)
    
    # Create a simple document
    doc1 = Document(
        page_content="This is my summary for DeepTSF platform.",
        metadata={
            "source": "..\RUN_7_18\deepTSF_doc.md",
            "author": "researcher",
            "date": "2024-01-15"
        }
    )
    
    print("Document content:", doc1.page_content)
    print("Document metadata:", doc1.metadata)
    print()
    
    # You can have multiple documents
    doc2 = Document(
        page_content="LightGBM on Short Term Load Forecasting papers.",
        metadata={
            "source": "..\RUN_7_25\STLF_reviews.md", 
            "topic": "STLF"
        }
    )
    
    documents = [doc1, doc2]
    print(f"We now have {len(documents)} documents")
    print()
    
    return documents

# =============================================================================
# 2. TEXT SPLITTING - Breaking Large Text into Chunks
# =============================================================================

def understand_text_splitting():
    """
    Why split text?
    - AI models have token limits (like GPT-3.5 has ~16k tokens)
    - Large documents need to be broken into smaller pieces
    - We want to keep related content together
    """
    
    print("✂️  Understanding Text Splitting")
    print("=" * 50)
    
    # Let's create a long document (like your markdown files)
    long_text = """
    # Energy Forecasting Research
    
    ## Introduction
    This research focuses on predicting energy consumption patterns using various machine learning approaches.
    
    ## Methodology
    We tested three main approaches:
    1. LSTM neural networks for sequential data
    2. ARIMA models for time series analysis  
    3. Random Forest for feature-based prediction
    
    ## Results
    The LSTM model achieved 95% accuracy on the test dataset.
    ARIMA performed well on seasonal patterns.
    Random Forest was best for incorporating weather data.
    
    ## Conclusion
    Ensemble methods combining all three approaches show the most promise.
    """
    
    doc = Document(page_content=long_text, metadata={"source": "full_research.md"})
    print(f"Original document length: {len(doc.page_content)} characters")
    print()
    
    # Basic text splitter
    basic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # Each chunk will be ~200 characters
        chunk_overlap=50,    # 50 characters overlap between chunks
        length_function=len  # How to measure length
    )
    
    chunks = basic_splitter.split_documents([doc])
    
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:100]}...")  # First 100 chars
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Metadata: {chunk.metadata}")
    
    return chunks

# =============================================================================  
# 3. SMART MARKDOWN SPLITTING
# =============================================================================

def understand_markdown_splitting():
    """
    For markdown files, we want to split by headers (# ## ###)
    This keeps related content together better than arbitrary character splits
    """
    
    print("🔍 Understanding Markdown Header Splitting")
    print("=" * 50)
    
    markdown_text = """
# Global Energy Forecasting Study

## Literature Review
Previous studies have shown various approaches to energy forecasting.
The most common methods include statistical and machine learning approaches.

### Statistical Methods
ARIMA and seasonal decomposition are widely used.
These methods work well for data with clear patterns.

### Machine Learning Methods  
Random Forest and SVM have gained popularity.
They can handle non-linear relationships better.

## Our Methodology
We propose a hybrid approach combining the best of both worlds.

### Data Collection
We collected 5 years of hourly energy consumption data.
Weather data was also incorporated as external factors.

### Model Training
Three separate models were trained and then combined.
"""

    # Markdown header splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),      # Main sections
            ("##", "Header 2"),     # Subsections  
            ("###", "Header 3"),    # Sub-subsections
        ]
    )
    
    # Split by headers
    header_chunks = markdown_splitter.split_text(markdown_text)
    
    print(f"Split markdown into {len(header_chunks)} chunks by headers:")
    for i, chunk in enumerate(header_chunks):
        print(f"\nChunk {i+1}:")
        if hasattr(chunk, 'page_content'):
            print(f"Content: {chunk.page_content[:150]}...")
            print(f"Metadata: {chunk.metadata}")
        else:
            print(f"Content: {chunk[:150]}...")
    
    return header_chunks

# =============================================================================
# 4. COMBINING BOTH STRATEGIES  
# =============================================================================

def understand_hierarchical_splitting():
    """
    Best practice: Split by headers first, then by size if chunks are still too big
    This is what we do in the RAG system
    """
    
    print("🏗️  Understanding Hierarchical Splitting")
    print("=" * 50)
    
    # Long markdown with big sections
    long_markdown = """
# Energy Forecasting Results

## LSTM Model Analysis
The LSTM model was trained on 5 years of data with the following architecture:
- Input layer: 24 time steps (24 hours)
- Hidden layers: 2 LSTM layers with 128 units each
- Dense layers: 2 fully connected layers with 64 and 32 units
- Output layer: Single neuron for energy prediction
- Activation: ReLU for hidden layers, linear for output
- Optimizer: Adam with learning rate 0.001
- Loss function: Mean Squared Error
- Batch size: 32
- Epochs: 100 with early stopping

The model performance was evaluated using:
- Mean Absolute Error (MAE): 2.3 kWh
- Root Mean Squared Error (RMSE): 3.1 kWh  
- Mean Absolute Percentage Error (MAPE): 4.2%
- R-squared: 0.94

Feature importance analysis revealed that:
1. Historical consumption (previous 24 hours) - 45% importance
2. Time of day - 25% importance
3. Day of week - 15% importance
4. Temperature - 10% importance
5. Other weather factors - 5% importance

The model showed excellent performance on weekday patterns but struggled slightly with holiday predictions.
"""

    # Step 1: Split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )
    
    header_chunks = markdown_splitter.split_text(long_markdown)
    
    # Step 2: Split large chunks further
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    
    final_chunks = []
    for chunk in header_chunks:
        if isinstance(chunk, Document):
            content = chunk.page_content
            metadata = chunk.metadata
        else:
            content = chunk
            metadata = {}
            
        # If chunk is too big, split it further
        if len(content) > 300:
            sub_chunks = text_splitter.split_text(content)
            for sub_chunk in sub_chunks:
                final_chunks.append(Document(
                    page_content=sub_chunk,
                    metadata={**metadata, "chunk_type": "sub_chunk"}
                ))
        else:
            final_chunks.append(Document(
                page_content=content,
                metadata={**metadata, "chunk_type": "header_chunk"}
            ))
    
    print(f"Final result: {len(final_chunks)} optimally-sized chunks")
    for i, chunk in enumerate(final_chunks):
        print(f"\nChunk {i+1} ({chunk.metadata.get('chunk_type', 'unknown')}):")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Preview: {chunk.page_content[:100]}...")
        if chunk.metadata:
            print(f"Headers: {chunk.metadata}")
    
    return final_chunks

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run all the examples to understand text processing"""
    
    print("🚀 LangChain Text Processing Tutorial")
    print("=" * 60)
    print()
    
    # Step 1: Basic documents
    docs = understand_documents()
    input("Press Enter to continue to text splitting...")
    print()
    
    # Step 2: Basic splitting  
    chunks = understand_text_splitting()
    input("Press Enter to continue to markdown splitting...")
    print()
    
    # Step 3: Markdown splitting
    md_chunks = understand_markdown_splitting()
    input("Press Enter to continue to hierarchical splitting...")
    print()
    
    # Step 4: Advanced hierarchical splitting
    final_chunks = understand_hierarchical_splitting()
    
    print("\n🎉 Congratulations! You now understand:")
    print("✅ What Documents are and why they're useful")
    print("✅ Why we need to split text into chunks")
    print("✅ How RecursiveCharacterTextSplitter works")
    print("✅ How MarkdownHeaderTextSplitter preserves structure")
    print("✅ How to combine both for optimal results")
    
    print("\n➡️  Next: We'll learn about embeddings and vector stores!")

if __name__ == "__main__":
    main()