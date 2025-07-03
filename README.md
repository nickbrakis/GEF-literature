# Global Energy Forecasting (GEF) Literature Repository

*Last updated: 2025-07-03*

## 🤖 AI-Generated Summary

This repository contains research notes and documentation for a thesis on Global Energy Forecasting. The automated summary will be generated once the GitHub Actions workflow is set up and runs.

---

## 📁 Repository Structure

This repository contains research notes and documentation organized by date in `RUN_XX_X` folders, where the format represents the research sessions.

Current folders:
- `RUN_20_6/` - Early research notes on clustering, energy load forecasting, and darts guide
- `RUN_27_6/` - Extended analysis including EDA, Enefit dataset documentation, and GEF papers
- `RUN_4_7/` - Latest pipeline documentation and dataset information

## 🔄 Automated Updates

This README is automatically updated whenever new markdown files are added to the repository using GitHub Actions and OpenAI API.

## 📝 Contributing

When adding new research notes:
1. Create markdown files in the appropriate `RUN_XX_X` folder
2. Use descriptive filenames
3. Follow consistent markdown formatting
4. The summary will be automatically updated on push

## 🛠️ Setup Instructions

To enable automated README generation:

1. **Set up OpenAI API Key**:
   - Go to your GitHub repository settings
   - Navigate to "Secrets and variables" → "Actions"
   - Add a new secret named `OPENAI_API_KEY` with your OpenAI API key

2. **Install dependencies** (for local testing):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally** (optional):
   ```bash
   python scripts/generate_summary.py
   ```

---
*This summary was generated automatically using AI. For detailed information, please refer to the individual markdown files in each folder.*
