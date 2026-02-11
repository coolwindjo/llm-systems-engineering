# Turing College AI Engineering Labs 🤖

Welcome to my repository for the Turing College AI Engineering program. This repository documents my journey, experiments, and code implementations as I progress through the curriculum, focusing on Large Language Models (LLMs), RAG systems, and AI Agents.

> **Note:** This repository contains lab exercises and experimental code. Solutions to graded projects may be withheld or strictly limited in accordance with academic integrity guidelines.

## 📚 Curriculum Overview

Based on the Turing College AI Engineering track, this repository is organized into the following sprints:

### Sprint 1: Fundamentals of LLMs & Prompt Engineering
*   **Focus:** Understanding LLM APIs (OpenAI), Prompt Engineering techniques, and basic evaluation.
*   **Key Concepts:** Tokenization, Temperature, System Prompts, In-context Learning.
*   **Labs:**
    *   Text generation basics
    *   Classification & Regression using Embeddings
    *   Zero-shot classification

### Sprint 2: RAG (Retrieval-Augmented Generation) & System Engineering
*   **Focus:** Building systems that ground LLM responses in external data to prevent hallucinations.
*   **Key Concepts:** Vector Databases, Embeddings, Semantic Search, Data Ingestion pipelines.
*   **Labs:**
    *   Semantic Search implementation
    *   Building a basic RAG pipeline
    *   Document chunking and indexing strategies

### Sprint 3: AI Agents & Advanced Workflows
*   **Focus:** Moving from linear chains to autonomous agents that can use tools and iterate.
*   **Key Concepts:** Tool use (Function Calling), LangGraph, Reasoning loops, Multi-agent systems.
*   **Labs:**
    *   Building a custom agent with tools
    *   Implementing agentic workflows with LangGraph

## 🛠️ Tech Stack & Tools

*   **Languages:** Python (Primary), TypeScript (Optional)
*   **Models:** OpenAI GPT-4o/GPT-4o-mini (via API)
*   **Frameworks:** LangChain, LangGraph
*   **Libraries:** Pandas, NumPy, Scikit-learn, Tiktoken
*   **Environment Management:** `uv` (Recommended) or `pip`/`conda`
*   **Editor:** VS Code

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+ installed
*   OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/turing-ai-engineering-labs.git
   cd turing-ai-engineering-labs
2. Set up a virtual environment I recommend using uv for faster package management, or standard venv.
3. Install dependencies
4. Environment Variables Create a .env file in the root directory and add your API key. Do not commit this file.

```
📂 Project Structure
.
├── sprint_1_prompt_engineering/
│   ├── notebooks/          # Jupyter notebooks for experiments
│   └── src/                # Python scripts for production-like code
├── sprint_2_rag/
│   ├── data/               # Sample datasets (e.g., food reviews, wikipedia chunks)
│   └── ...
├── sprint_3_agents/
│   └── ...
├── .gitignore
├── requirements.txt
└── README.md
```
💡 Key Learnings & Insights
• Shift to System Engineering: Moving beyond simple prompts to designing robust systems where LLMs act as the kernel process (2026-02-11 Stand-up).
• Deep Understanding: Avoiding "Vibe Coding" (copy-pasting AI code without understanding). Focusing on debugging and understanding the logic behind every line (2026-02-11 Stand-up).
• Evaluation: Implementing rigorous testing to measure model performance beyond just "feeling" it works.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

--------------------------------------------------------------------------------
Created by [Your Name] during the Turing College AI Engineering Batch [Batch Number, e.g., 2026-02].
