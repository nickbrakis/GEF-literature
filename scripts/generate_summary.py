#!/usr/bin/env python3
"""
AI-powered README generator for GEF Literature Repository
Reads all markdown files and generates a comprehensive summary using OpenAI API
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import openai
from openai import OpenAI

def find_markdown_files(directory: Path) -> List[Path]:
    """
    Recursively find all .md files in the directory, excluding README.md
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of Path objects for found markdown files
    """
    markdown_files = []
    
    for item in directory.rglob("*.md"):
        # Skip hidden directories and the root README.md
        if (not any(part.startswith('.') for part in item.parts) and 
            item.name != "README.md"):
            markdown_files.append(item)
    
    return sorted(markdown_files)

def read_markdown_files(files: List[Path], base_dir: Path) -> List[Dict]:
    """
    Read and structure content from markdown files
    
    Args:
        files: List of markdown file paths
        base_dir: Base directory for relative path calculation
        
    Returns:
        List of dictionaries containing file information and content
    """
    content = []
    
    for file_path in files:
        try:
            relative_path = file_path.relative_to(base_dir)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            content.append({
                'path': str(relative_path),
                'content': file_content,
                'folder': str(relative_path.parent),
                'size': len(file_content)
            })
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return content

def generate_summary(markdown_content: List[Dict]) -> str:
    """
    Generate summary using OpenAI API
    
    Args:
        markdown_content: List of file content dictionaries
        
    Returns:
        Generated summary text
    """
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare content for analysis with smart truncation for GPT-3.5-turbo
    content_text = ""
    total_chars = 0
    # GPT-3.5-turbo has 16K tokens, roughly 4 chars per token
    # Leave room for prompt and response (about 10K tokens for content)
    max_chars = 30000  # Conservative limit for GPT-3.5-turbo
    
    # Sort files by size (smaller first to include more variety)
    sorted_files = sorted(markdown_content, key=lambda x: x['size'])
    
    for file_info in sorted_files:
        file_header = f"## File: {file_info['path']}\n\n"
        file_footer = "\n\n---\n\n"
        file_content = file_info['content']
        
        # Calculate space needed for this file
        full_file_size = len(file_header) + len(file_content) + len(file_footer)
        
        # If we can fit the whole file, add it
        if total_chars + full_file_size <= max_chars:
            content_text += file_header + file_content + file_footer
            total_chars += full_file_size
        else:
            # Try to fit a truncated version
            remaining_space = max_chars - total_chars - len(file_header) - len(file_footer) - 50
            if remaining_space > 200:  # Only if we have meaningful space left
                content_text += file_header
                content_text += file_content[:remaining_space]
                content_text += "\n\n[File truncated due to length limits...]"
                content_text += file_footer
                break
            else:
                break
    
    if total_chars >= max_chars:
        print(f"⚠️  Content truncated from {sum(f['size'] for f in markdown_content):,} to {total_chars:,} chars for GPT-3.5-turbo limits.")
    
    prompt = f"""Analyze markdown files from a Global Energy Forecasting (GEF) thesis research repository.

The repository contains research notes (mostly in Greek), literature reviews, and analysis in date-organized folders (RUN_XX_X format).

Create a comprehensive summary with:
1. **Overview**: Research focus and objectives
2. **Key Topics**: Main research areas and methodologies
3. **Folder Structure**: What each folder/run contains
4. **Research Progress**: Timeline and evolution
5. **Key Findings**: Important insights, datasets, methodologies
6. **Literature & References**: Key papers, tools, frameworks

Use proper markdown formatting. Be concise but informative.

Files analyzed: {len(markdown_content)} | Content: {sum(f['size'] for f in markdown_content):,} chars

{content_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant specialized in creating comprehensive summaries of academic research documentation. Focus on extracting key insights, methodologies, and research progress."
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
        error_msg = str(e)
        if "context_length_exceeded" in error_msg or "maximum context length" in error_msg:
            print(f"⚠️  Still hitting token limits. Trying with even more aggressive truncation...")
            # Try again with much smaller content
            return generate_summary_fallback(markdown_content, client)
        else:
            print(f"Error generating summary: {e}")
            return None

def generate_summary_fallback(markdown_content: List[Dict], client) -> str:
    """
    Fallback function with very aggressive truncation for GPT-3.5-turbo
    """
    print("Using fallback strategy with minimal content...")
    
    # Create a very short summary of each file
    file_summaries = []
    for file_info in markdown_content:
        # Take only first 500 chars of each file
        content_preview = file_info['content'][:500]
        file_summaries.append(f"**{file_info['path']}**: {content_preview}...")
    
    # Limit to first 10 files to stay within token limits
    if len(file_summaries) > 10:
        file_summaries = file_summaries[:10]
        file_summaries.append("...[Additional files omitted due to length limits]")
    
    content_text = "\n\n".join(file_summaries)
    
    prompt = f"""Analyze these markdown files from a Global Energy Forecasting (GEF) research repository.

Create a brief summary covering:
1. Research focus and objectives
2. Key methodologies mentioned
3. Main findings and insights

Files: {len(markdown_content)} total
Content previews:

{content_text}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Create a concise research summary from the provided file previews."
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
        print(f"Fallback also failed: {e}")
        return f"Unable to generate AI summary due to content length limitations. Repository contains {len(markdown_content)} markdown files with research notes about Global Energy Forecasting."

def create_readme_content(summary: str, file_count: int, total_size: int) -> str:
    """
    Create the complete README.md content
    
    Args:
        summary: AI-generated summary
        file_count: Number of markdown files processed
        total_size: Total size of content processed
        
    Returns:
        Complete README.md content
    """
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    return f"""# Global Energy Forecasting (GEF) Literature Repository

*Last updated: {timestamp}*
*Files analyzed: {file_count} | Content size: {total_size:,} characters*

## 🤖 AI-Generated Summary

{summary}

---

## 📁 Repository Structure

This repository contains research notes and documentation organized by date in `RUN_XX_X` folders, where the format represents the research sessions.

## 🔄 Automated Updates

This README is automatically updated whenever new markdown files are added to the repository using GitHub Actions and OpenAI API.

## 📝 Contributing

When adding new research notes:
1. Create markdown files in the appropriate `RUN_XX_X` folder
2. Use descriptive filenames
3. Follow consistent markdown formatting
4. The summary will be automatically updated on push

## 🛠️ Technical Details

- **Summary Generator**: Python script using OpenAI models
- **Automation**: GitHub Actions workflow
- **Last Analysis**: {file_count} markdown files processed
- **Content Volume**: {total_size:,} characters analyzed

---
*This summary was generated automatically using AI. For detailed information, please refer to the individual markdown files in each folder.*"""

def main():
    """Main execution function"""
    try:
        print("🔍 Finding markdown files...")
        
        # Get the repository root directory
        repo_root = Path.cwd()
        markdown_files = find_markdown_files(repo_root)
        
        print(f"Found {len(markdown_files)} markdown files")
        
        if not markdown_files:
            print("No markdown files found. Creating basic README...")
            basic_summary = "No research notes found yet. Add some markdown files to see an AI-generated summary here!"
            readme_content = create_readme_content(basic_summary, 0, 0)
            
            with open(repo_root / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            return
        
        print("📖 Reading file contents...")
        markdown_content = read_markdown_files(markdown_files, repo_root)
        total_size = sum(item['size'] for item in markdown_content)
        
        print(f"📊 Total content: {total_size:,} characters from {len(markdown_content)} files")
        
        print("🤖 Generating AI summary...")
        summary = generate_summary(markdown_content)
        
        if not summary:
            print("❌ Failed to generate summary")
            sys.exit(1)
        
        print("📝 Creating README.md...")
        readme_content = create_readme_content(summary, len(markdown_content), total_size)
        
        with open(repo_root / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ README.md updated successfully!")
        
        # Print some statistics
        print(f"\n📈 Statistics:")
        print(f"   • Files processed: {len(markdown_content)}")
        print(f"   • Total content: {total_size:,} characters")
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
            print(f"   • {folder}: {len(files)} files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
