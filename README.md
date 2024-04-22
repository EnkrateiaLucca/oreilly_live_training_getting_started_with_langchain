# Notebooks for the O'Reilly live-training: "Getting Started with LangChain"

- [Live-training official website from O'Reilly](https://learning.oreilly.com/live-events/getting-started-with-langchain/0636920098586/0636920098585/)
# Overview


## Notebooks

1. **Getting Started with LangChain**
   - [Introduction to LangChain](https://colab.research.google.com/github/your-username/your-repo/blob/main/notebooks/1.0-intro-to-langchain.ipynb)
   - [LangChain Application Development Structure](https://colab.research.google.com/github/your-username/your-repo/blob/main/notebooks/1.1-langchain-app-dev-structure.ipynb)

2. **Composing and Utilizing Chains**
   - [Composing Chain Pipelines with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.0-LCEL-interface-composing-chains.ipynb)
   - [Integrating OpenAI Functions and Runnables with Tools](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.1-langchain-openai-functions-runnables-with-tools.ipynb)
   - [Utilizing Pydantic with OpenAI in LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.2-langchain-pydantic-openai.ipynb)
   - [Learning Chain with ADEPT Example](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.3-learning-chain-adept-example.ipynb)
   - [Extraction and Tagging with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.4-extraction-tagging-with-langchain.ipynb)

3. **Advanced Query and Dynamic Content**
   - [QA with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/3.0-qa-with-langchain.ipynb)
   - [Querying CSV Data with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/3.1-langchain-query-csv.ipynb)
   - [Dynamic Quiz over PDF](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/3.2-dynamic-quiz-over-pdf.ipynb)

4. **Building Intelligent Agents**
   - [Building LLM Agents with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/4.0-building-llm-agents-with-langchain.ipynb)
   - [Enhancing LLM Agents with Memory](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/4.1-llm-agents-with-memory.ipynb)
   - [Prototype GitHub Agent with LangChain](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/4.2-langchain-github-agent-prototype.ipynb)

5. **Demonstrations and Practical Applications**
   - [Research Workflows](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/5.0-demos-research-workflows.ipynb)
   - [Automated Article Scraping with Playwright](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/5.1-demo-playwright-scrape-articles.py)

6. **Deployment and Integration**
   - [LangChain Deployment Recipe](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/6.0-langchain-deploy-recipe.ipynb)


## Setup

**Conda**

- Install [anaconda](https://www.anaconda.com/download)
- Create an environment: `conda create -n oreilly-langchain python=3.11`
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