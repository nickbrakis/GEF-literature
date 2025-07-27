#!/usr/bin/env python3
# type: ignore
"""
LangChain Tutorial Step 2: Understanding Embeddings and Vector Stores
The core of semantic search and RAG systems
"""

import os
import numpy as np
from langchain.schema import Document 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================================================================
# 1. WHAT ARE EMBEDDINGS? 
# =============================================================================

def understand_embeddings_concept():
    """
    Embeddings convert text into numbers (vectors) that capture meaning.
    Similar texts get similar numbers!
    
    Think of it like GPS coordinates for meaning:
    - "energy forecasting" and "power prediction" would be close
    - "energy forecasting" and "cooking recipes" would be far apart
    """
    
    print("🧠 Understanding Embeddings (The Magic Behind Semantic Search)")
    print("=" * 70)
    
    print("What are embeddings?")
    print("- They convert text into lists of numbers (vectors)")
    print("- Each number represents some aspect of meaning")
    print("- Similar texts get similar vectors")
    print("- This lets computers understand 'meaning', not just keywords")
    print()
    
    print("Example (simplified):")
    print("Text: 'energy forecasting'     → Vector: [0.2, 0.8, 0.1, 0.9, ...]")
    print("Text: 'power prediction'       → Vector: [0.3, 0.7, 0.2, 0.8, ...]  ← Similar!")
    print("Text: 'cooking pasta'          → Vector: [0.9, 0.1, 0.7, 0.2, ...]  ← Different!")
    print()
    
    print("Real OpenAI embeddings have 1,536 dimensions (numbers)!")
    print("Each dimension captures some aspect of meaning we can't easily understand.")
    print()

def create_sample_embeddings():
    """
    Let's create real embeddings and see how they work
    Note: You need OPENAI_API_KEY set for this to work
    """
    
    print("🔢 Creating Real Embeddings")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Skipping real embeddings demo.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return None
    
    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Sample texts related to your research
    sample_texts = [
        "LSTM neural networks for energy consumption forecasting",
        "Deep learning models predict power usage patterns",  # Similar to above
        "ARIMA time series analysis for seasonal energy data",
        "Statistical methods for electricity demand prediction",  # Similar to above
        "Weather data correlation with energy consumption",
        "Temperature effects on power grid load balancing",  # Similar to above
        "How to cook pasta with marinara sauce",  # Very different!
    ]
    
    print("Creating embeddings for sample texts...")
    try:
        # Get embeddings for all texts
        vectors = embeddings_model.embed_documents(sample_texts)
        
        print(f"✅ Created {len(vectors)} embeddings")
        print(f"Each embedding has {len(vectors[0])} dimensions")
        print()
        
        # Show similarity between related texts
        def calculate_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors"""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        print("🔍 Similarity Analysis:")
        print("-" * 40)
        
        # Compare similar texts
        sim1 = calculate_similarity(vectors[0], vectors[1])  # LSTM vs Deep learning
        print(f"LSTM vs Deep learning similarity: {sim1:.3f}")
        
        sim2 = calculate_similarity(vectors[2], vectors[3])  # ARIMA vs Statistical
        print(f"ARIMA vs Statistical methods similarity: {sim2:.3f}")
        
        sim3 = calculate_similarity(vectors[4], vectors[5])  # Weather vs Temperature
        print(f"Weather vs Temperature similarity: {sim3:.3f}")
        
        # Compare very different texts
        sim4 = calculate_similarity(vectors[0], vectors[6])  # LSTM vs Cooking
        print(f"LSTM vs Cooking pasta similarity: {sim4:.3f}")
        
        print()
        print("💡 Notice: Similar topics have higher similarity scores!")
        print("   This is how semantic search works!")
        
        return vectors, sample_texts
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return None

# =============================================================================
# 2. VECTOR STORES - Where Embeddings Live
# =============================================================================

def understand_vector_stores():
    """
    Vector stores are databases optimized for storing and searching vectors.
    Think of them as super-fast similarity search engines.
    """
    
    print("🗄️  Understanding Vector Stores")
    print("=" * 50)
    
    print("What is a vector store?")
    print("- A special database for storing embedding vectors")
    print("- Optimized for fast similarity search")
    print("- Can quickly find 'similar' documents")
    print("- Much faster than comparing every document manually")
    print()
    
    print("Popular vector stores:")
    print("- FAISS: Facebook's fast similarity search (what we use)")
    print("- Pinecone: Cloud-based vector database")
    print("- Chroma: Open-source embedding database")
    print("- Weaviate: Vector search engine")
    print()
    
    print("How it works:")
    print("1. Store: Put all your document embeddings in the database")
    print("2. Query: Convert your question to an embedding")
    print("3. Search: Find the most similar document embeddings")
    print("4. Retrieve: Get back the original documents")
    print()

def create_vector_store_demo():
    """
    Create a real vector store with sample research documents
    """
    
    print("🏗️  Creating a Vector Store")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Skipping vector store demo.")
        return None
    
    # Create sample research documents
    research_docs = [
        Document(
            page_content="Our LSTM model achieved 94% accuracy in predicting hourly energy consumption. The model used 3 hidden layers with 128 units each.",
            metadata={"source": "lstm_results.md", "folder": "RUN_01", "topic": "deep_learning"}
        ),
        Document(
            page_content="ARIMA(2,1,2) model showed excellent performance on seasonal energy data with MAPE of 3.2%.",
            metadata={"source": "arima_analysis.md", "folder": "RUN_02", "topic": "time_series"}
        ),
        Document(
            page_content="Weather data correlation analysis revealed temperature as the strongest predictor of energy demand with r=0.87.",
            metadata={"source": "weather_correlation.md", "folder": "RUN_03", "topic": "correlation"}
        ),
        Document(
            page_content="Random Forest ensemble with 100 trees outperformed individual models in cross-validation testing.",
            metadata={"source": "ensemble_methods.md", "folder": "RUN_04", "topic": "machine_learning"}
        ),
        Document(
            page_content="Data preprocessing involved handling missing values, outlier detection, and feature scaling using StandardScaler.",
            metadata={"source": "data_preprocessing.md", "folder": "RUN_05", "topic": "preprocessing"}
        ),
    ]
    
    print(f"📚 Created {len(research_docs)} sample research documents")
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Create vector store
        print("🔄 Creating FAISS vector store...")
        vector_store = FAISS.from_documents(research_docs, embeddings)
        
        print("✅ Vector store created successfully!")
        print(f"   Stored {len(research_docs)} documents with embeddings")
        print()
        
        return vector_store, research_docs
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None

# =============================================================================
# 3. SEMANTIC SEARCH - The Magic in Action
# =============================================================================

def demonstrate_semantic_search(vector_store, docs):
    """
    Show how semantic search finds relevant documents
    """
    
    print("🔍 Semantic Search Demo")
    print("=" * 50)
    
    if vector_store is None:
        print("❌ No vector store available for demo")
        return
    
    # Test queries that show semantic understanding
    test_queries = [
        "deep learning neural network performance",  # Should find LSTM doc
        "statistical time series forecasting accuracy",  # Should find ARIMA doc  
        "temperature weather impact on energy",  # Should find weather doc
        "data cleaning and preparation steps",  # Should find preprocessing doc
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 40)
        
        try:
            # Perform similarity search
            similar_docs = vector_store.similarity_search(query, k=2)  # Get top 2 matches
            
            for i, doc in enumerate(similar_docs, 1):
                print(f"  {i}. Source: {doc.metadata['source']}")
                print(f"     Folder: {doc.metadata['folder']}")
                print(f"     Content: {doc.page_content[:100]}...")
                print()
                
        except Exception as e:
            print(f"❌ Search error: {e}")

def demonstrate_search_with_scores(vector_store):
    """
    Show similarity scores to understand how good the matches are
    """
    
    print("📊 Search with Similarity Scores")
    print("=" * 50)
    
    if vector_store is None:
        print("❌ No vector store available for demo")
        return
    
    query = "machine learning model performance evaluation"
    print(f"🔎 Query: '{query}'")
    print()
    
    try:
        # Search with scores
        docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
        
        print("Results (lower score = more similar):")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Source: {doc.metadata['source']}")
            print(f"     Content: {doc.page_content[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Search error: {e}")

# =============================================================================
# 4. PUTTING IT ALL TOGETHER
# =============================================================================

def complete_workflow_demo():
    """
    Show the complete workflow: Documents → Chunks → Embeddings → Vector Store → Search
    """
    
    print("🔄 Complete RAG Workflow Demo")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Skipping complete workflow.")
        return
    
    # Step 1: Large document (simulating your markdown files)
    large_doc_content = """
# Energy Forecasting Research Results

## LSTM Deep Learning Approach
Our LSTM neural network model was trained on 5 years of hourly energy consumption data.
The architecture consisted of 3 LSTM layers with 128, 64, and 32 units respectively.
We used dropout regularization of 0.3 to prevent overfitting.
The model achieved a final MAPE of 2.8% on the test set.

## ARIMA Statistical Model
We implemented ARIMA(2,1,2) for comparison with the deep learning approach.
The model was fitted using maximum likelihood estimation.
Seasonal decomposition was applied first to handle yearly patterns.
ARIMA achieved MAPE of 4.1%, showing good but inferior performance to LSTM.

## Feature Engineering
Temperature data was the most important external feature.
We created lag features for the previous 24 and 48 hours.
Day of week and hour of day were encoded as categorical features.
Holiday indicators were added to handle special consumption patterns.

## Model Ensemble
Final predictions combined LSTM (60%) and ARIMA (40%) using weighted averaging.
The ensemble approach reduced MAPE to 2.3%, our best result.
Cross-validation confirmed the robustness of this approach.
"""
    
    print("📄 Step 1: Create large document")
    large_doc = Document(
        page_content=large_doc_content,
        metadata={"source": "complete_research.md", "folder": "FINAL_RUN"}
    )
    print(f"   Document length: {len(large_doc.page_content)} characters")
    
    # Step 2: Split into chunks
    print("\n✂️  Step 2: Split into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    
    chunks = splitter.split_documents([large_doc])
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings and vector store
    print("\n🧠 Step 3: Create embeddings and vector store")
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"   ✅ Vector store created with {len(chunks)} embeddings")
        
        # Step 4: Test semantic search
        print("\n🔍 Step 4: Test semantic search")
        
        test_queries = [
            "What was the LSTM model architecture?",
            "How did ARIMA perform compared to deep learning?",
            "What features were most important?",
        ]
        
        for query in test_queries:
            print(f"\n   Query: {query}")
            results = vector_store.similarity_search(query, k=1)
            if results:
                print(f"   Best match: {results[0].page_content[:150]}...")
            
        print("\n🎉 Complete workflow successful!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run all embedding and vector store demos"""
    
    print("🚀 LangChain Embeddings & Vector Stores Tutorial")
    print("=" * 60)
    print()
    
    # Step 1: Understand the concept
    understand_embeddings_concept()
    input("Press Enter to see real embeddings...")
    print()
    
    # Step 2: Create real embeddings
    result = create_sample_embeddings()
    input("Press Enter to learn about vector stores...")
    print()
    
    # Step 3: Understand vector stores
    understand_vector_stores()
    input("Press Enter to create a vector store...")
    print()
    
    # Step 4: Create vector store
    store_result = create_vector_store_demo()
    if store_result:
        vector_store, docs = store_result
        input("Press Enter to see semantic search...")
        print()
        
        # Step 5: Demonstrate semantic search
        demonstrate_semantic_search(vector_store, docs)
        input("Press Enter to see search scores...")
        print()
        
        demonstrate_search_with_scores(vector_store)
        input("Press Enter for complete workflow demo...")
        print()
    
    # Step 6: Complete workflow
    complete_workflow_demo()
    
    print("\n🎉 Congratulations! You now understand:")
    print("✅ What embeddings are and why they capture meaning")
    print("✅ How vector stores work as similarity search engines")
    print("✅ How semantic search finds relevant content")
    print("✅ The complete RAG workflow: Docs → Chunks → Embeddings → Search")
    
    print("\n➡️  Next: We'll learn about Retrieval and LLM Chains!")

if __name__ == "__main__":
    main()