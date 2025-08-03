#!/usr/bin/env python3
# type: ignore
"""
LangChain Tutorial Step 3: Retrieval Chains and LLMs
Where vector search meets language models for intelligent Q&A
"""

import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler

# =============================================================================
# 1. UNDERSTANDING LLMs IN LANGCHAIN
# =============================================================================

def understand_llms():
    """
    LLMs (Large Language Models) are the 'brain' that generates answers.
    LangChain makes it easy to work with different LLM providers.
    """
    
    print("🧠 Understanding LLMs in LangChain")
    print("=" * 50)
    
    print("What are LLMs in LangChain?")
    print("- The 'brain' that generates human-like text")
    print("- LangChain supports many providers: OpenAI, Anthropic, Hugging Face, etc.")
    print("- You can easily switch between different models")
    print("- They take context + question → generate answer")
    print()
    
    print("Popular LLM options:")
    print("- gpt-3.5-turbo: Fast, cost-effective, good for most tasks")
    print("- gpt-3.5-turbo-16k: Longer context window (16k tokens)")
    print("- gpt-4: Most capable, slower, more expensive")
    print("- gpt-4-32k: Longest context window")
    print()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Set it to try LLM examples.")
        return None
    
    # Initialize different LLMs
    print("🔧 Initializing LLMs...")
    
    try:
        # Basic LLM
        basic_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,  # Low temperature = more focused, consistent answers
        )
        
        # LLM with streaming (shows response as it's generated)
        streaming_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        print("✅ LLMs initialized successfully!")
        print(f"   Basic LLM model: {basic_llm.model_name}")
        print(f"   Temperature: {basic_llm.temperature} (lower = more focused)")
        
        return basic_llm, streaming_llm
        
    except Exception as e:
        print(f"❌ Error initializing LLMs: {e}")
        return None

def test_basic_llm(llm):
    """Test LLM with a simple question"""
    
    if llm is None:
        return
    
    print("\n🧪 Testing Basic LLM")
    print("=" * 50)
    
    try:
        # Simple question without context
        question = "What is energy forecasting?"
        print(f"Question: {question}")
        print("Answer:")
        
        response = llm.invoke(question)
        print(response)
        
        print("\n💡 Notice: This is general knowledge, not from your research documents!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# =============================================================================
# 2. UNDERSTANDING RETRIEVERS
# =============================================================================

def understand_retrievers():
    """
    Retrievers are the bridge between vector stores and LLMs.
    They fetch relevant documents based on a query.
    """
    
    print("🔍 Understanding Retrievers")
    print("=" * 50)
    
    print("What is a Retriever?")
    print("- Connects vector stores to LLMs")
    print("- Takes a question → finds relevant documents")
    print("- Provides context to the LLM")
    print("- Different search strategies available")
    print()
    
    print("Retriever types:")
    print("- similarity: Find most similar documents (most common)")
    print("- mmr: Maximum Marginal Relevance (diverse results)")
    print("- similarity_score_threshold: Only results above certain similarity")
    print()
    
    print("Key parameters:")
    print("- k: How many documents to retrieve (e.g., k=3 gets top 3)")
    print("- search_kwargs: Additional search parameters")
    print()

def create_sample_retriever():
    """Create a sample retriever with research documents"""
    
    print("🏗️  Creating Sample Retriever")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Skipping retriever demo.")
        return None
    
    # Create sample research documents (your markdown files would be like this)
    research_docs = [
        Document(
            page_content="""
# LSTM Model Results
Our LSTM neural network achieved excellent performance on energy forecasting.
Architecture: 3 layers with 128, 64, 32 units. Used dropout 0.3 for regularization.
Training data: 5 years of hourly consumption data.
Results: MAPE 2.8%, RMSE 45.2 kWh, R² 0.94.
The model handled seasonal patterns well but struggled with holiday anomalies.
            """.strip(),
            metadata={"source": "lstm_results.md", "folder": "RUN_06", "topic": "deep_learning"}
        ),
        
        Document(
            page_content="""
# ARIMA Time Series Analysis
Statistical ARIMA(2,1,2) model for energy consumption prediction.
Applied seasonal decomposition first to handle yearly cycles.
Model fitted using maximum likelihood estimation method.
Performance: MAPE 4.1%, good for seasonal data but limited for complex patterns.
Works best when combined with other methods in ensemble approach.
            """.strip(),
            metadata={"source": "arima_analysis.md", "folder": "RUN_04", "topic": "time_series"}
        ),
        
        Document(
            page_content="""
# Feature Engineering Study
Analyzed importance of different features for energy prediction models.
Temperature correlation: r=0.87 (strongest predictor).
Created lag features: previous 24h and 48h consumption values.
Categorical encoding: day of week, hour of day, holiday indicators.
Feature selection reduced dimensionality from 45 to 12 key features.
            """.strip(),
            metadata={"source": "feature_engineering.md", "folder": "RUN_05", "topic": "preprocessing"}
        ),
        
        Document(
            page_content="""
# Model Ensemble Results  
Combined LSTM and ARIMA using weighted averaging approach.
Weights determined by cross-validation: LSTM 60%, ARIMA 40%.
Ensemble performance: MAPE 2.3% (best achieved).
Ensemble more robust than individual models, handles various data patterns.
Recommended approach for production deployment.
            """.strip(),
            metadata={"source": "ensemble_results.md", "folder": "RUN_07", "topic": "ensemble"}
        ),
    ]
    
    try:
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(research_docs, embeddings)
        
        # Create different types of retrievers
        retrievers = {
            "basic": vector_store.as_retriever(),
            "top_3": vector_store.as_retriever(search_kwargs={"k": 3}),
            "diverse": vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": 3, "fetch_k": 6}  # Fetch 6, return 3 most diverse
            )
        }
        
        print(f"✅ Created vector store with {len(research_docs)} documents")
        print("✅ Created 3 different retrievers")
        
        return vector_store, retrievers, research_docs
        
    except Exception as e:
        print(f"❌ Error creating retriever: {e}")
        return None

def test_retrievers(retrievers):
    """Test different retriever strategies"""
    
    if not retrievers:
        return
    
    print("\n🧪 Testing Different Retrievers")
    print("=" * 50)
    
    query = "What was the performance of deep learning models?"
    print(f"Query: '{query}'")
    print()
    
    for name, retriever in retrievers.items():
        print(f"📋 {name.upper()} RETRIEVER:")
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"   Retrieved {len(docs)} documents:")
            
            for i, doc in enumerate(docs, 1):
                print(f"   {i}. {doc.metadata['source']} (topic: {doc.metadata['topic']})")
                print(f"      Preview: {doc.page_content[:100]}...")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 30)

# =============================================================================
# 3. PROMPT TEMPLATES - CONTROLLING LLM BEHAVIOR
# =============================================================================

def understand_prompt_templates():
    """
    Prompt templates control how the LLM processes retrieved information.
    They're like instructions for the AI.
    """
    
    print("📝 Understanding Prompt Templates")
    print("=" * 50)
    
    print("What are Prompt Templates?")
    print("- Instructions that tell the LLM how to behave")
    print("- Include placeholders for dynamic content (context, question)")
    print("- Critical for getting good results from RAG")
    print("- Can customize for different use cases")
    print()
    
    print("Key components:")
    print("- System role: Who is the AI? (research assistant, expert, etc.)")
    print("- Context: The retrieved documents")
    print("- Question: The user's question")
    print("- Instructions: How to format the answer")
    print()

def create_custom_prompts():
    """Create different prompt templates for different purposes"""
    
    print("🛠️  Creating Custom Prompt Templates")
    print("=" * 50)
    
    # Basic Q&A prompt
    basic_qa_prompt = PromptTemplate(
        template="""Use the following context to answer the question. If you don't know the answer based on the context, say you don't know.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Research-specific prompt
    research_prompt = PromptTemplate(
        template="""You are an expert research assistant analyzing Global Energy Forecasting literature.

Based on the following research documents, answer the question with:
1. Direct information from the documents
2. Specific metrics, results, or findings mentioned
3. Which files/studies the information comes from
4. Any limitations or gaps in the available information

Research Context:
{context}

Question: {question}

Detailed Research Answer:""",
        input_variables=["context", "question"]
    )
    
    # Summary generation prompt
    summary_prompt = PromptTemplate(
        template="""Analyze the following research documents and create a comprehensive summary.

Focus on:
- Main research objectives and approaches
- Key findings and performance metrics
- Methodologies used
- Important insights and conclusions

Documents:
{context}

Research Summary:""",
        input_variables=["context"]
    )
    
    print("✅ Created 3 different prompt templates:")
    print("   1. Basic Q&A: Simple question answering")
    print("   2. Research-focused: Detailed analysis with citations")
    print("   3. Summary: Comprehensive document summarization")
    
    return {
        "basic": basic_qa_prompt,
        "research": research_prompt,
        "summary": summary_prompt
    }

# =============================================================================
# 4. RETRIEVAL CHAINS - PUTTING IT ALL TOGETHER
# =============================================================================

def understand_retrieval_chains():
    """
    Retrieval chains combine retrievers + LLMs + prompts into a complete system.
    This is the core of RAG!
    """
    
    print("⛓️  Understanding Retrieval Chains")
    print("=" * 50)
    
    print("What is a Retrieval Chain?")
    print("- Combines Retriever + LLM + Prompt Template")
    print("- The complete RAG pipeline in one object")
    print("- Input: Question → Output: Answer with sources")
    print()
    
    print("How it works:")
    print("1. Question comes in")
    print("2. Retriever finds relevant documents")
    print("3. Documents become 'context' in prompt")
    print("4. LLM generates answer based on context")
    print("5. Return answer + source documents")
    print()
    
    print("Chain types:")
    print("- stuff: Put all retrieved docs in one prompt (most common)")
    print("- map_reduce: Summarize each doc separately, then combine")
    print("- refine: Iteratively refine answer with each document")
    print()

def create_retrieval_chains(vector_store, llm, prompts):
    """Create different types of retrieval chains"""
    
    if not vector_store or not llm or not prompts:
        print("❌ Missing components for chain creation")
        return None
    
    print("🔗 Creating Retrieval Chains")
    print("=" * 50)
    
    try:
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create different chains with different prompts
        chains = {}
        
        # Basic Q&A chain
        chains["basic"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompts["basic"]},
            return_source_documents=True
        )
        
        # Research-focused chain
        chains["research"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            chain_type_kwargs={"prompt": prompts["research"]},
            return_source_documents=True
        )
        
        print("✅ Created retrieval chains:")
        print("   1. Basic Q&A chain")
        print("   2. Research-focused chain")
        
        return chains
        
    except Exception as e:
        print(f"❌ Error creating chains: {e}")
        return None

def test_retrieval_chains(chains):
    """Test the retrieval chains with sample questions"""
    
    if not chains:
        return
    
    print("\n🧪 Testing Retrieval Chains")
    print("=" * 50)
    
    test_questions = [
        "What was the best performing model and its accuracy?",
        "How do LSTM and ARIMA models compare?",
        "What features are most important for energy forecasting?",
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("=" * 60)
        
        # Test with research chain (more detailed)
        try:
            result = chains["research"]({"query": question})
            
            print("🤖 ANSWER:")
            print(result["result"])
            
            print("\n📚 SOURCES:")
            for i, doc in enumerate(result.get("source_documents", []), 1):
                print(f"   {i}. {doc.metadata['source']} (folder: {doc.metadata['folder']})")
                print(f"      {doc.page_content[:100]}...")
                print()
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("Press Enter for next question...")

# =============================================================================
# 5. COMPLETE RAG WORKFLOW DEMO
# =============================================================================

def complete_rag_demo():
    """Demonstrate the complete RAG workflow from start to finish"""
    
    print("🚀 Complete RAG Workflow Demo")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Cannot run complete demo.")
        return
    
    print("This simulates processing your actual markdown files...")
    
    # Simulate a large research document
    large_research = """
# Comprehensive Energy Forecasting Study

## Executive Summary
This study evaluates multiple approaches to energy consumption forecasting using 5 years of data.
We tested statistical, machine learning, and deep learning methods.

## Dataset Description
- Time period: 2018-2023 (5 years)
- Frequency: Hourly measurements
- Total records: 43,800 data points
- External factors: Temperature, humidity, day type
- Missing data: 0.3% (handled via interpolation)

## Methodology Comparison

### LSTM Deep Learning
- Architecture: 3-layer LSTM (128, 64, 32 units)
- Training: 80% train, 10% validation, 10% test
- Optimization: Adam optimizer, learning rate 0.001
- Regularization: Dropout 0.3, early stopping
- Results: MAPE 2.8%, RMSE 45.2 kWh, Training time: 2.5 hours

### ARIMA Statistical Model
- Model: ARIMA(2,1,2) with seasonal component
- Seasonal period: 24 hours and 168 hours (weekly)
- Parameter estimation: Maximum likelihood
- Results: MAPE 4.1%, RMSE 62.8 kWh, Training time: 15 minutes

### Random Forest Ensemble
- Trees: 100 estimators
- Features: 15 engineered features including lags
- Cross-validation: 5-fold
- Results: MAPE 3.5%, RMSE 51.3 kWh, Training time: 30 minutes

## Feature Importance Analysis
1. Previous 24h consumption: 35% importance
2. Temperature: 28% importance  
3. Hour of day: 18% importance
4. Day of week: 12% importance
5. Previous 48h consumption: 7% importance

## Final Ensemble Model
Combined LSTM (60%) and ARIMA (40%) using weighted averaging.
Weights optimized through cross-validation.
Final performance: MAPE 2.3%, RMSE 42.1 kWh.
This represents a 17% improvement over the best individual model.

## Deployment Recommendations
The ensemble model is recommended for production use.
Real-time inference capability with sub-second predictions.
Model retraining recommended monthly to maintain accuracy.
"""
    
    try:
        # Step 1: Create document and split
        print("📄 Step 1: Processing large research document...")
        doc = Document(
            page_content=large_research,
            metadata={"source": "comprehensive_study.md", "folder": "FINAL_RESULTS"}
        )
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents([doc])
        print(f"   Split into {len(chunks)} chunks")
        
        # Step 2: Create vector store
        print("\n🧠 Step 2: Creating embeddings and vector store...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("   ✅ Vector store created")
        
        # Step 3: Setup LLM and chain
        print("\n🤖 Step 3: Setting up LLM and retrieval chain...")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
        
        research_prompt = PromptTemplate(
            template="""You are analyzing energy forecasting research. Use the context to provide detailed, accurate answers.

Context: {context}

Question: {question}

Provide a comprehensive answer with specific metrics and findings from the research:""",
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": research_prompt},
            return_source_documents=True
        )
        
        print("   ✅ RAG chain ready")
        
        # Step 4: Interactive Q&A
        print("\n🔍 Step 4: Interactive Q&A Demo")
        
        sample_questions = [
            "What was the best performing model and its exact performance metrics?",
            "How much data was used and what was the time period?",
            "What are the most important features for prediction?",
            "What are the deployment recommendations?",
        ]
        
        for question in sample_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 60)
            
            result = qa_chain({"query": question})
            print("🤖 Answer:")
            print(result["result"])
            print()
            
            input("Press Enter for next question...")
        
        print("\n🎉 Complete RAG workflow successful!")
        print("This is exactly how your system will work with your research files!")
        
    except Exception as e:
        print(f"❌ Error in complete demo: {e}")

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run all retrieval chain and LLM demos"""
    
    print("🚀 LangChain Retrieval Chains & LLMs Tutorial")
    print("=" * 60)
    print()
    
    # Step 1: Understand LLMs
    llm_result = understand_llms()
    if llm_result:
        basic_llm, streaming_llm = llm_result
        test_basic_llm(basic_llm)
    input("Press Enter to learn about retrievers...")
    print()
    
    # Step 2: Understand retrievers
    understand_retrievers()
    retriever_result = create_sample_retriever()
    if retriever_result:
        vector_store, retrievers, docs = retriever_result
        test_retrievers(retrievers)
    input("Press Enter to learn about prompt templates...")
    print()
    
    # Step 3: Prompt templates
    understand_prompt_templates()
    prompts = create_custom_prompts()
    input("Press Enter to learn about retrieval chains...")
    print()
    
    # Step 4: Retrieval chains
    understand_retrieval_chains()
    if llm_result and retriever_result:
        chains = create_retrieval_chains(vector_store, basic_llm, prompts)
        if chains:
            test_retrieval_chains(chains)
    input("Press Enter for complete RAG demo...")
    print()
    
    # Step 5: Complete workflow
    complete_rag_demo()
    
    print("\n🎉 Congratulations! You now understand:")
    print("✅ How LLMs work in LangChain")
    print("✅ What retrievers do and how they work")
    print("✅ How prompt templates control LLM behavior")
    print("✅ How retrieval chains combine everything")
    print("✅ The complete RAG workflow from question to answer")
    
    print("\n➡️  Next: We'll put it all together in your custom RAG system!")

if __name__ == "__main__":
    main()