# ReAct Agent Implementation

A Python implementation of a ReAct (Reasoning and Acting) agent that synergizes reasoning and acting in language models. This project is inspired by the [ReAct paper](https://react-lm.github.io/) from ICLR 2023, aiming to combine the reasoning capabilities of language models with action execution using tools and assistants.

## Table of Contents

-   [Features](#features)
-   [Installation](#installation)
-   [Getting Started](#getting-started)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [References](#references)
-   [License](#license)

## Features

-   **Unified API Interface**: Interacts seamlessly with OpenAI GPT models and Ollama models through a unified API.
-   **Reasoning and Acting**: Uses reflection to decide on actions and interacts with tools to perform tasks.
-   **Tool Integration**: Supports tools like SQL database queries and mathematical calculations.
-   **Assistants**: Can decompose complex questions into simpler sub-questions for recursive solving.
-   **Memory Management**: Maintains a simple memory of past interactions for context.
-   **Extensibility**: Designed to be easily extended with additional tools and assistants.

## Installation

### Prerequisites

-   Python 3.7 or higher
-   An OpenAI API key (if using OpenAI models)
-   SQLite database file (e.g., `sql_lite_database.db`)

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/react-agent.git
    cd react-agent
    ```

2. **Set Up LLM**

    For OpenAI models: create a `.env` file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

    For Ollama models: set up the Ollama server and modify the `OLLAMA_MODEL` variable in the `react_agent.py` script with the model you want to use.

## Usage

Run the agent by executing the main script:

```bash
python react_agent.py
```

### Example

In the `react_agent.py` script, you can modify the `question` variable to ask different questions:

```python
question = "How did sales vary between Q1 and Q2 of 2024 in percentage and amount? Use the decomposition tool once."
agent.run_agent(question)
agent.save_context_to_html("agent_context.html")
agent.save_memory()
```

### Output

The agent will output a trace of its reasoning and actions, and save the context to an HTML file for easier viewing. It will also save its memory of interactions for future sessions.

## Project Structure

-   **`react_agent.py`**: Main script to run the ReAct agent.
-   **`prompts.py`**: Holds the system prompts used for reflection and action determination.
-   **`requirements.txt`**: Lists all Python dependencies.
-   **`agent_memory.json`**: JSON file where the agent's memory is stored.
-   **`agent_context.html`**: HTML file containing the agent's reasoning and actions in a readable format.

## References

-   **ReAct Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io/)
    -   _Authors_: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
    -   _Conference_: ICLR 2023
-   **Original ReAct Repository**: [https://github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct)

## License

This project has no license.

Feel free to use it as you want.

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.
