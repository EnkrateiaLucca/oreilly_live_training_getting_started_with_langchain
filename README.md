# Notebooks for the O'Reilly live-training: "Getting Started with LangChain"

- [Live-training official website from O'Reilly](https://learning.oreilly.com/live-events/getting-started-with-langchain/0636920098586/0636920098585/)
# Overview


## Notebooks

1. [Introduction to LangChain](notebooks/1.0-intro-to-langchain.ipynb)
2. [Composing Chain Pipelines with LangChain](notebooks/2.0-composing-chain-pipelines-with-langchain.ipynb)
3. [Introduction to LangChain Expression Language (LCEL)](notebooks/2.1-LCEL-interface.ipynb)
4. [LangChain OpenAI Functions and Runnables with Tools](notebooks/2.2-langchain-openai-functions-runnables-with-tools.ipynb)
5. [QA with LangChain](notebooks/3.0-qa-with-langchain.ipynb)
6. [LangChain Query CSV](notebooks/3.1-langchain-query-csv.ipynb)
7. [LangChain QA with GPT-4](notebooks/3.2-langchain-qa-gpt4.ipynb)
8. [LangChain QA with CSV and PDF](notebooks/3.3-langchain-qa-csv-pdf.ipynb)
9. [Building LLM Agents with LangChain](notebooks/4.0-building-llm-agents-with-langchain.ipynb)
10. [Conversational Agent with LangChain](notebooks/4.1-conversational-agent-with-langchain.ipynb)
11. [LangChain GitHub Agent Prototype](notebooks/4.2-langchain-github-agent-prototype.ipynb)

## Setup

**Conda**

- Install [anaconda](https://www.anaconda.com/download)
- Create an environment: `conda create -n oreilly-langchain`
- Activate your environment with: `conda activate oreilly-langchain`
- Install requirements with: `pip install -r requirements.txt`
- Setup your openai [API key](https://platform.openai.com/)

**Pip**


1. **Create a Virtual Environment:**
    Navigate to your project directory. If using Python 3's built-in `venv`:
    ```bash
    python -m venv oreilly_env
    ```
    If you're using `virtualenv`:
    ```bash
    virtualenv oreilly_env
    ```

2. **Activate the Virtual Environment:**
    - **On Windows:**
      ```bash
      .\oreilly_env\Scripts\activate
      ```
    - **On macOS and Linux:**
      ```bash
      source oreilly_env/bin/activate
      ```

3. **Install Dependencies from `requirements.txt`:**
    ```bash
    pip install python-dotenv
    pip install -r requirements.txt
    ```

4. Setup your openai [API key](https://platform.openai.com/)

Remember to deactivate the virtual environment once you're done by simply typing:
```bash
deactivate
```

## Setup your .env file

- Change the `.env.example` file to `.env` and add your OpenAI API key.

## To use this Environment with Jupyter Notebooks:

```python3 -m ipykernel install --user --name=oreilly-langchain```

## Official Training Website

For more information about the live-training, visit the [official website](https://learning.oreilly.com/live-events/getting-started-with-langchain/0636920098586/0636920098585/).