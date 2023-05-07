# SearchGPT

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
---

SearchGPT is a tool-using agent that utilizes the power of GPT-3.5 to provide search-based responses to user queries. This project is based on the [How to build a tool-using agent with LangChain](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb) example from the OpenAI Cookbook, with the addition of an interface using Gradio library.

## Usage

You need to obtain API keys and put them into `.env` file:
```
OPENAI_API_KEY=sk-***
PINECONE_API_KEY=***
PINECONE_ENVIRONMENT=northamerica-northeast1-gcp
SERPAPI_API_KEY=***
```

To download and process podcasts run:
```sh
python scripts/pinecone_podcasts.py
```

After that to start the SearchGPT interface, run the following command:
```sh
gradio searchgpt
```

This will launch the Gradio web interface for the SearchGPT chatbot.
