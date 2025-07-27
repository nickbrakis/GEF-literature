#!/usr/bin/env python3
"""
AI-powered README generator for GEF Literature Repository
Enhanced version with robust token management and chunking strategies
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import openai
from openai import OpenAI

# Token limits for different models (conservative estimates)
MODEL_LIMITS = {
    "gpt-3.5-turbo": {"max_tokens": 4096, "chars_per_token": 4},
    "gpt-3.5-turbo-16k": {"max_tokens": 16384, "chars_per_token": 4},
    "gpt-4": {"max_tokens": 8192, "chars_per_token": 4},
    "gpt-4-32k": {"max_tokens": 32768, "chars_per_token": 4}
}

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo-16k") -> int:
    """Estimate token count for a given text"""
    chars_per_token = MODEL_LIMITS.get(model, {}).get("chars_per_token", 4)
    return len(text) // chars_per_token

def find_markdown_files(directory: Path) -> List[Path]:
    """
    Recursively find all .md files in the directory, excluding README.md
    """
    markdown_files = []
    
    for item in directory.rglob("*.md"):
        if (not any(part.startswith('.') for part in item.parts) and 
            item.name != "README.md"):
            markdown_files.append(item)
    
    return sorted(markdown_files)

def extract_key_content(content: str, max_chars: int = 2000) -> str:
    """
    Extract key content from markdown files using smart truncation
    Prioritizes headers, key phrases, and structured content
    """
    if len(content) <= max_chars:
        return content
    
    lines = content.split('\n')
    extracted_lines = []
    current_length = 0
    
    # Priority order: headers, bullet points, numbered lists, then regular content
    priority_patterns = [
        r'^#{1,6}\s+',  # Headers
        r'^\s*[-*+]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^\s*>\s+',  # Blockquotes
        r'^\s*\|.*\|',  # Tables
    ]
    
    # First pass: collect high-priority content
    for line in lines:
        if current_length >= max_chars * 0.7:  # Use 70% for priority content
            break
            
        is_priority = any(re.match(pattern, line) for pattern in priority_patterns)
        
        if is_priority and line.strip():
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length <= max_chars * 0.7:
                extracted_lines.append(line)
                current_length += line_length
    
    # Second pass: add regular content to fill remaining space
    remaining_space = max_chars - current_length
    if remaining_space > 100:  # Only if meaningful space remains
        regular_content = []
        for line in lines:
            if (line.strip() and 
                not any(re.match(pattern, line) for pattern in priority_patterns) and
                line not in extracted_lines):
                regular_content.append(line)
        
        # Add regular content until we hit the limit
        for line in regular_content:
            line_length = len(line) + 1
            if current_length + line_length <= max_chars - 50:  # Leave buffer
                extracted_lines.append(line)
                current_length += line_length
            else:
                break
    
    result = '\n'.join(extracted_lines)
    if len(result) < len(content):
        result += "\n\n[Content truncated...]"
    
    return result

def read_markdown_files(files: List[Path], base_dir: Path) -> List[Dict]:
    """
    Read and structure content from markdown files with smart content extraction
    """
    content = []
    
    for file_path in files:
        try:
            relative_path = file_path.relative_to(base_dir)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Extract key content instead of full content
            key_content = extract_key_content(file_content, max_chars=1500)
            
            content.append({
                'path': str(relative_path),
                'content': key_content,
                'original_size': len(file_content),
                'processed_size': len(key_content),
                'folder': str(relative_path.parent),
            })
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return content

def create_file_chunks(markdown_content: List[Dict], model: str = "gpt-3.5-turbo-16k") -> List[List[Dict]]:
    """
    Split files into chunks that fit within model token limits
    """
    max_tokens = MODEL_LIMITS[model]["max_tokens"]
    # Reserve tokens for prompt, response, and safety buffer
    available_tokens = max_tokens - 3000  # Conservative buffer
    max_chars = available_tokens * MODEL_LIMITS[model]["chars_per_token"]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Sort by folder to keep related content together
    sorted_content = sorted(markdown_content, key=lambda x: (x['folder'], x['path']))
    
    for file_info in sorted_content:
        file_size = len(f"## File: {file_info['path']}\n\n{file_info['content']}\n\n---\n\n")
        
        if current_size + file_size > max_chars and current_chunk:
            # Start new chunk
            chunks.append(current_chunk)
            current_chunk = [file_info]
            current_size = file_size
        else:
            current_chunk.append(file_info)
            current_size += file_size
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_chunk_summary(chunk: List[Dict], chunk_num: int, total_chunks: int, client) -> str:
    """
    Generate summary for a single chunk of files
    """
    content_text = ""
    for file_info in chunk:
        content_text += f"## File: {file_info['path']}\n\n"
        content_text += file_info['content']
        content_text += "\n\n---\n\n"
    
    prompt = f"""Analyze this chunk ({chunk_num}/{total_chunks}) of markdown files from a Global Energy Forecasting (GEF) research repository.

Focus on:
1. Key research topics and methodologies in these files
2. Important findings or insights
3. Folder/run organization patterns
4. Datasets or tools mentioned
5. Literature references

Files in this chunk: {len(chunk)}
Folders covered: {', '.join(set(f['folder'] for f in chunk))}

{content_text}

Provide a structured summary focusing on the most important research content."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are analyzing research documentation. Focus on key insights, methodologies, and research progress. Be concise but comprehensive."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating chunk {chunk_num} summary: {e}")
        return f"Chunk {chunk_num}: Analysis failed - contains files from {', '.join(set(f['folder'] for f in chunk))}"

def combine_chunk_summaries(chunk_summaries: List[str], total_files: int, client) -> str:
    """
    Combine multiple chunk summaries into a final comprehensive summary
    """
    combined_text = "\n\n".join([f"## Chunk {i+1} Summary:\n{summary}" 
                                for i, summary in enumerate(chunk_summaries)])
    
    prompt = f"""Combine these chunk summaries into a comprehensive overview of a Global Energy Forecasting (GEF) research repository.

Create a unified summary with:
1. **Overview**: Research focus and main objectives
2. **Key Topics**: Main research areas and methodologies
3. **Folder Structure**: Organization and what each contains
4. **Research Progress**: Timeline and evolution visible
5. **Key Findings**: Important insights and discoveries
6. **Literature & References**: Key papers, tools, frameworks mentioned

Total files analyzed: {total_files}

Chunk summaries to combine:
{combined_text}

Create a cohesive, well-structured summary that eliminates redundancy and highlights the most important aspects."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize multiple research summaries into a comprehensive, well-organized overview. Eliminate redundancy and focus on key insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2500,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error combining summaries: {e}")
        return "Multiple research chunks analyzed but final combination failed. Repository contains research on Global Energy Forecasting with various methodologies and findings."

def generate_summary(markdown_content: List[Dict]) -> str:
    """
    Generate summary using chunking strategy for large content
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Determine if we need chunking
    total_chars = sum(len(f"## File: {f['path']}\n\n{f['content']}\n\n---\n\n") for f in markdown_content)
    
    print(f"Total processed content: {total_chars:,} characters")
    
    # If content is small enough, process in one go
    if total_chars < 20000:  # Safe limit for single processing
        return generate_single_summary(markdown_content, client)
    else:
        return generate_chunked_summary(markdown_content, client)

def generate_single_summary(markdown_content: List[Dict], client) -> str:
    """Generate summary for small content that fits in one request"""
    content_text = ""
    for file_info in markdown_content:
        content_text += f"## File: {file_info['path']}\n\n"
        content_text += file_info['content']
        content_text += "\n\n---\n\n"
    
    prompt = f"""Analyze markdown files from a Global Energy Forecasting (GEF) research repository.

Create a comprehensive summary with:
1. **Overview**: Research focus and objectives
2. **Key Topics**: Main research areas and methodologies  
3. **Folder Structure**: What each folder/run contains
4. **Research Progress**: Timeline and evolution
5. **Key Findings**: Important insights, datasets, methodologies
6. **Literature & References**: Key papers, tools, frameworks

Files analyzed: {len(markdown_content)}

{content_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "Create comprehensive research summaries focusing on key insights and methodologies."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2500,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Single summary generation failed: {e}")
        return f"Repository contains {len(markdown_content)} research files about Global Energy Forecasting. Summary generation failed due to technical limitations."

def generate_chunked_summary(markdown_content: List[Dict], client) -> str:
    """Generate summary using chunking strategy for large content"""
    print("🔄 Using chunked processing strategy...")
    
    # Create chunks
    chunks = create_file_chunks(markdown_content)
    print(f"📦 Created {len(chunks)} chunks")
    
    # Generate summary for each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"🤖 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} files)...")
        summary = generate_chunk_summary(chunk, i+1, len(chunks), client)
        chunk_summaries.append(summary)
    
    # Combine all chunk summaries
    print("🔗 Combining chunk summaries...")
    final_summary = combine_chunk_summaries(chunk_summaries, len(markdown_content), client)
    
    return final_summary

def create_readme_content(summary: str, file_count: int, processed_size: int, original_size: int) -> str:
    """Create the complete README.md content"""
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    return f"""# Global Energy Forecasting (GEF) Literature Repository

*Last updated: {timestamp}*
*Files analyzed: {file_count} | Original content: {original_size:,} chars | Processed: {processed_size:,} chars*

## 🤖 AI-Generated Summary

{summary}

---

## 📁 Repository Structure

This repository contains research notes and documentation organized by date in `RUN_XX_X` folders, where the format represents the research sessions.

## 🔄 Automated Updates

This README is automatically updated whenever new markdown files are added to the repository using GitHub Actions and OpenAI API with intelligent content processing.

## 📝 Contributing

When adding new research notes:
1. Create markdown files in the appropriate `RUN_XX_X` folder
2. Use descriptive filenames and clear headers
3. Follow consistent markdown formatting
4. The summary will be automatically updated on push

## 🛠️ Technical Details

- **Summary Generator**: Python script using OpenAI models with chunking strategy
- **Content Processing**: Smart extraction focusing on headers, lists, and key content
- **Automation**: GitHub Actions workflow
- **Last Analysis**: {file_count} markdown files processed
- **Processing Efficiency**: {processed_size:,} chars analyzed from {original_size:,} original chars

---
*This summary was generated automatically using AI with intelligent content chunking and processing. For detailed information, please refer to the individual markdown files in each folder.*"""

def main():
    """Main execution function"""
    try:
        print("🔍 Finding markdown files...")
        
        repo_root = Path.cwd()
        markdown_files = find_markdown_files(repo_root)
        
        print(f"Found {len(markdown_files)} markdown files")
        
        if not markdown_files:
            print("No markdown files found. Creating basic README...")
            basic_summary = "No research notes found yet. Add some markdown files to see an AI-generated summary here!"
            readme_content = create_readme_content(basic_summary, 0, 0, 0)
            
            with open(repo_root / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            return
        
        print("📖 Reading and processing file contents...")
        markdown_content = read_markdown_files(markdown_files, repo_root)
        
        processed_size = sum(item['processed_size'] for item in markdown_content)
        original_size = sum(item['original_size'] for item in markdown_content)
        
        print(f"📊 Content processed: {processed_size:,} chars from {original_size:,} original chars")
        print(f"📈 Compression ratio: {(processed_size/original_size)*100:.1f}%")
        
        print("🤖 Generating AI summary...")
        summary = generate_summary(markdown_content)
        
        if not summary:
            print("❌ Failed to generate summary")
            sys.exit(1)
        
        print("📝 Creating README.md...")
        readme_content = create_readme_content(summary, len(markdown_content), processed_size, original_size)
        
        with open(repo_root / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ README.md updated successfully!")
        
        # Print statistics
        print(f"\n📈 Statistics:")
        print(f"   • Files processed: {len(markdown_content)}")
        print(f"   • Original content: {original_size:,} characters")
        print(f"   • Processed content: {processed_size:,} characters")
        print(f"   • Compression ratio: {(processed_size/original_size)*100:.1f}%")
        print(f"   • Summary length: {len(summary):,} characters")
        
        # Show files by folder
        folders = {}
        for item in markdown_content:
            folder = item['folder']
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(item['path'])
        
        print(f"\n📂 Files by folder:")
        for folder, files in sorted(folders.items()):
            original_chars = sum(item['original_size'] for item in markdown_content if item['folder'] == folder)
            processed_chars = sum(item['processed_size'] for item in markdown_content if item['folder'] == folder)
            print(f"   • {folder}: {len(files)} files ({processed_chars:,}/{original_chars:,} chars)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()