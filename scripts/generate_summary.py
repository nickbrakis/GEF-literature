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
    
    # Prepare content for analysis
    content_text = ""
    for file_info in markdown_content:
        content_text += f"## File: {file_info['path']}\n\n"
        content_text += f"{file_info['content']}\n\n"
        content_text += "---\n\n"
    
    # Check if content is too long and truncate if necessary
    max_chars = 50000  # Conservative limit for GPT-4
    if len(content_text) > max_chars:
        print(f"⚠️  Content too long ({len(content_text)} chars). Truncating to {max_chars} chars.")
        content_text = content_text[:max_chars] + "\n\n[Content truncated...]"
    
    prompt = f"""You are analyzing markdown files from a thesis research repository about Global Energy Forecasting (GEF). 

The repository contains research notes, literature reviews, and analysis organized in folders with dates (RUN_XX_X format).

Please analyze all the provided markdown files and create a comprehensive summary that includes:

1. **Overview**: A brief description of the research focus and objectives
2. **Key Topics Covered**: Main research areas and methodologies discussed
3. **Folder Structure**: Brief description of what each folder/run contains
4. **Research Progress**: Timeline and evolution of the research
5. **Key Findings**: Important insights, datasets, and methodologies mentioned
6. **Literature & References**: Key papers, tools, or frameworks referenced

Please make the summary well-structured, informative, and suitable for a README.md file. Use proper markdown formatting.

Files analyzed: {len(markdown_content)} markdown files
Total content size: {sum(f['size'] for f in markdown_content):,} characters

Here are the markdown files to analyze:

{content_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
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
        print(f"Error generating summary: {e}")
        return None

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

- **Summary Generator**: Python script using OpenAI GPT-4
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
