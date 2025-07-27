import os
# Check if API key is available
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  OPENAI_API_KEY not set. Skipping real embeddings demo.")
    print("   Set your API key: export OPENAI_API_KEY='your-key-here'")