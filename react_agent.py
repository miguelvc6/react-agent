import json
import sqlite3
import textwrap
from typing import List
import os

from ansi2html import Ansi2HTMLConverter
import openai
import ollama
from pydantic import BaseModel, ValidationError

REFLECTION_SYSTEM_PROMPT = """GENERAL INSTRUCTIONS
Your task is to reflect on the question and context to decide how to solve it.
You must decide whether to use a tool, an assistant, or give the final answer if you have sufficient information.
Write a brief reflection with the indicated response format.
Do not call any actions or tools, return only the reflection.

AVAILABLE TOOLS
- list_sql_tables: {"Description": "Returns a list with the names of tables present in the database", "Arguments": None}
- sql_db_schema: {"Description": "Returns the schema of a specific table in the database", "Arguments": table_name - str}
- sql_db_query: {"Description": "Executes an SQL query in the sqlite3 database and returns the results. \
    Do not use without first observing the table schema", "Arguments": sql_query - str}
- math_calculator: {"Description": "Performs basic mathematical calculations", "Arguments": expression - str}

AVAILABLE ASSISTANTS
- decomposition: {"Description": "Divides a complex question into simpler sub-parts and calls agents \
    to solve them recursively. Use only for complex questions", "Arguments": question - str}

AVAILABLE ACTION
- final_answer: {"Description": "Final answer for the user. Must answer the question asked.", "Arguments": "answer - str"}

RESPONSE FORMAT
REFLECTION >> <Fill>
"""

ACTION_SYSTEM_PROMPT_01 = """GENERAL INSTRUCTIONS
Your task is to answer questions using an SQL database and performing mathematical calculations.
If you already have enough information, you should provide a final answer.
You must decide whether to use a tool, an assistant, or give the final answer, and return a response following the response format.
Fill with null where no tool or assistant is required.

IMPORTANT:
- The response must be in valid JSON format.
- Ensure all text strings are properly escaped.
- Do not include line breaks within strings.
- If the argument is an SQL query or a mathematical expression, include it on a single line and in double quotes.

AVAILABLE TOOLS
- list_sql_tables: {"Description": "Returns a list with the names of tables present in the database", "Arguments": null}
- sql_db_schema: {"Description": "Returns the schema of a specific table in the database", "Arguments": "table_name" - str}
- sql_db_query: {"Description: "Executes an SQL query in the sqlite3 database and returns the results. \
    Do not use without first observing the table schema", Arguments: sql_query - str}
- math_calculator: {"Description": "Performs basic mathematical calculations", "Arguments": "expression" - str}
"""
ACTION_SYSTEM_PROMPT_DECOMPOSITION = """
AVAILABLE ASSISTANTS
- decomposition: {"Description: "Divides a complex question into simpler sub-parts and calls agents \
    to solve them recursively. Use only for complex questions", Arguments: question - str}
"""

ACTION_SYSTEM_PROMPT_02 = """
AVAILABLE ACTION
- final_answer: {"Description": "Final answer for the user. Must answer the question asked.", "Arguments": "answer - str"}

RESPONSE FORMAT
{
  "request": "<Fill>",
  "argument": "<Fill or null>"
}

EXAMPLES:

1. Using a tool without an argument:
{
  "request": "list_sql_tables",
  "argument": null
}

2. Using a tool with an argument:
{
  "request": "sql_db_schema",
  "argument": "ORDERS"
}

3. Using sql_db_query with an SQL query:
{
  "request": "sql_db_query",
  "argument": "SELECT * FROM ORDERS WHERE date(ORD_DATE) BETWEEN date('2024-01-01') AND date('2024-06-30');"
}

4. Final answer:
{
  "request": "final_answer",
  "argument": "Sales varied by 15% between Q1 and Q2 of 2024."
}
"""


class UnifiedChatAPI:
    def __init__(self, model="gpt-4o-mini", openai_api_key=None):
        self.model = model
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def set_api(self, api):
        """Sets the API to either 'openai' or 'ollama'."""
        self.api = api.lower()

    def chat(self, messages):
        """Wrapper for chat API. Switches between OpenAI and Ollama APIs based on configuration."""
        if self.api == "openai":
            return self._openai_chat(messages)
        elif self.api == "ollama":
            return self._ollama_chat(messages)
        else:
            raise ValueError("Unsupported API. Please set the API to 'openai' or 'ollama'.")

    def _openai_chat(self, messages):
        response = openai.ChatCompletion.create(model=self.model, messages=messages, api_key=self.api_key)
        return response.choices[0].message.content

    def _ollama_chat(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


class SimpleMemory:
    def __init__(self):
        self.question_trace = []
        self.answer_trace = []

    def add_interaction(self, question, answer):
        self.question_trace.append(question)
        self.answer_trace.append(answer)

    def get_context(self):
        if not self.question_trace:
            return ""
        else:
            context_lines = []
            for q, a in zip(self.question_trace, self.answer_trace):
                context_lines.append(f"QUESTION: {q}")
                context_lines.append(f"ANSWER: {a}")
            return "\n".join(context_lines)


class DecomposedQuestion(BaseModel):
    sub_questions: List[str]


class AgentAction(BaseModel):
    request: str
    argument: str | None


class AnswersSummary(BaseModel):
    summary: str


class AgenteReAct:
    def __init__(self, model="gpt-4o-mini", db_path="./sql_lite_database.db", memory_path="agent_memory.json"):
        """Initialize AgentCore with database path."""
        self.model = model
        self.memory = SimpleMemory()
        self.context = ""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        try:
            self._connect_db()
        except RuntimeError as e:
            print(f"Failed to connect to database at {db_path}: {e}")
        self.memory_path = memory_path

    # DB Management
    def _connect_db(self):
        """Connect to the database."""
        if not os.path.exists(self.db_path):
            raise RuntimeError(f"Database file not found at: {self.db_path}")

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            self._close_db()
            raise RuntimeError(f"Database connection failed: {e}")

    def _close_db(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.cursor = None
        self.conn = None

    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self._close_db()

    # Memory Management
    def save_memory(self):
        """Save the agent memory to a JSON file."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(
                {"question_trace": self.memory.question_trace, "answer_trace": self.memory.answer_trace}, f, indent=4
            )

    # Agent Reflections
    def reflection(self, question: str) -> dict:
        """Perform an agent reflection."""

        if not self.context:
            context = "<No previous questions have been asked>"
        else:
            context = self.context
        agent_template = f"""CONTEXTUAL INFORMATION
        {context}

        QUESTION
        {question}"""

        agent_reflection = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": agent_template},
            ],
        )
        return agent_reflection.choices[0].message.content

    # Agent Action
    def action(self, question: str, recursion=False, max_retrials: int = 3) -> dict:
        """Perform an agent action."""
        action_system_prompt = (
            ACTION_SYSTEM_PROMPT_01 + (not recursion) * ACTION_SYSTEM_PROMPT_DECOMPOSITION + ACTION_SYSTEM_PROMPT_02
        )

        if not self.context:
            context = "<No se han hecho preguntas anteriores>"
        else:
            context = self.context
        agent_template = f"""CONTEXTUAL INFORMATION
        {context}

        QUESTION
        {question}"""

        for attempt in range(max_retrials):
            agent_action_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": action_system_prompt},
                    {"role": "user", "content": agent_template},
                ],
            )

            assistant_reply = agent_action_response.choices[0].message.content

            try:
                agent_action = json.loads(assistant_reply)
                validated_response = AgentAction.model_validate(agent_action)
                return validated_response
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")
                # Provide feedback to the assistant about the error
                agent_template += (
                    "\n\nERROR >> The response is not valid JSON or does not follow the expected format."
                    "Please ensure the response is in the correct format."
                )
                continue

        raise RuntimeError("Maximum number of retries reached without successful validation.")

    def run_agent(self, question: str, recursion: bool = False, indent_level: int = 0) -> str:
        """Runs the ReAct agent to answer a question"""
        if not recursion:
            self.context = self.memory.get_context()
            print("\n")

        while True:
            try:
                # Perform reflection
                reflection = self.reflection(question=question)
                indent = "    " * indent_level
                reflection_msg = (
                    f"{indent}{self.color_text('REFLECTION >> ', 'REFLECTION')}"
                    f"{textwrap.fill(reflection.split('>> ')[1], width=100)}\n".replace(
                        "\n", "\n" + (indent_level + 1) * "\t"
                    )[: -(indent_level + 1)]
                )
                self.context += reflection_msg

                # Decide on an action
                try:
                    step = self.action(question=question, recursion=recursion)
                except json.JSONDecodeError as e:
                    error_msg = (
                        f"{indent}{self.color_text('ERROR >> ', 'ERROR')}"
                        f"{textwrap.fill(f"Failed to parse the assistant's response as JSON: {str(e)}", width=100)}\n".replace(
                            "\n", "\n" + (indent_level + 1) * "\t"
                        )[
                            : -(indent_level + 1)
                        ]
                    )
                    self.context += error_msg
                    continue

                os.system("cls" if os.name == "nt" else "clear")
                print(self.context)

                # Append action and argument to context
                action_msg = (
                    f"{indent}{self.color_text('ACTION >> ', 'ACTION')}"
                    f"{textwrap.fill(step.request, width=100)}\n".replace("\n", "\n" + (indent_level + 1) * "\t")[
                        : -(indent_level + 1)
                    ]
                )
                self.context += action_msg
                if step.argument:
                    arg_msg = (
                        f"{indent}{self.color_text('ARGUMENT >> ', 'ARGUMENT')}"
                        f"{textwrap.fill(step.argument, width=100)}\n".replace("\n", "\n" + (indent_level + 1) * "\t")[
                            : -(indent_level + 1)
                        ]
                    )
                    self.context += arg_msg

                result = None

                # Execute the chosen action
                try:
                    if step.request == "list_sql_tables":
                        result = self.list_sql_tables()
                    elif step.request == "sql_db_schema":
                        result = self.sql_db_schema(step.argument)
                    elif step.request == "sql_db_query":
                        result = self.sql_db_query(step.argument)
                    elif step.request == "math_calculator":
                        result = self.math_calculator(step.argument)
                    elif step.request == "decomposition":
                        result = self.decompose_question(question=step.argument)
                        return_msg = str(result).replace("\n", "\n\t")
                        obs_msg = (
                            f"{indent}{self.color_text('OBSERVATION >> ', 'OBSERVATION')}"
                            f"{textwrap.fill(return_msg, width=100)}\n".replace(
                                "\n", "\n" + (indent_level + 1) * "\t"
                            )[: -(indent_level + 1)]
                        )
                        self.context += obs_msg

                        # Answer subquestions recursively
                        answers = []
                        for subquestion in result.sub_preguntas:
                            subq_msg = (
                                f"{indent}{self.color_text('SUBQUESTION >> ', 'SUBQUESTION')}"
                                f"{textwrap.fill(subquestion, width=100)}\n".replace(
                                    "\n", "\n" + (indent_level + 1) * "\t"
                                )[: -(indent_level + 1)]
                            )
                            self.context += subq_msg
                            self.run_agent(subquestion, recursion=True, indent_level=min(indent_level + 1, 3))

                            # Extract answer from contextcm
                            if "FINAL ANSWER" in self.context:
                                subquestion_answer = self.context.split("FINAL ANSWER >> ")[-1].strip().split("\n")[0]
                                answers.append(subquestion_answer)

                        # Summarize answers
                        summary = self.answers_summarizer(result.sub_preguntas, answers)
                        self.context += (
                            f"GENERATED RESPONSE TO SUBQUESTIONS >> "
                            f"{textwrap.fill(summary.resumen, width=100)}\n".replace(
                                "\n", "\n" + (indent_level + 1) * "\t"
                            )[: -(indent_level + 1)]
                        )
                        continue

                    # Manage final answer
                    elif step.request == "final_answer":
                        # Update memory
                        self.memory.add_interaction(question, step.argument)
                        final_answer_msg = (
                            f"{indent}{self.color_text('FINAL ANSWER >> ', 'FINAL ANSWER')}"
                            f"{textwrap.fill(step.argument, width=100)}\n".replace(
                                "\n", "\n" + (indent_level + 1) * "\t"
                            )[: -(indent_level + 1)]
                        )
                        self.context += final_answer_msg
                        os.system("cls" if os.name == "nt" else "clear")
                        print(self.context)
                        return step.argument

                    # Manage tool execution errors
                    if result is None:
                        raise ValueError(f"Could not execute {step.request} with the provided argument.")

                    # Append observation to context
                    return_msg = str(result).replace("\n", "\n\t")
                    obs_msg = (
                        f"{indent}{self.color_text('OBSERVATION >> ', 'OBSERVATION')}"
                        f"{textwrap.fill(return_msg, width=100)}\n".replace("\n", "\n" + (indent_level + 1) * "\t")[
                            : -(indent_level + 1)
                        ]
                    )
                    self.context += obs_msg

                except Exception as e:
                    # Append error observation to context
                    error_msg = (
                        f"{indent}{self.color_text('OBSERVATION >> ', 'OBSERVATION')}"
                        f"{textwrap.fill(f'Error ejecutando {step.request}: {str(e)}', width=100)}\n".replace(
                            "\n", "\n" + (indent_level + 1) * "\t"
                        )[: -(indent_level + 1)]
                    )
                    self.context += error_msg
                    continue

            except Exception as e:
                error_msg = (
                    f"{indent}{self.color_text('ERROR >> ', 'ERROR')}"
                    f"{textwrap.fill(str(e), width=100)}\n".replace("\n", "\n" + (indent_level + 1) * "\t")[
                        : -(indent_level + 1)
                    ]
                )
                self.context += error_msg
                continue

    # Text Formatting and Output
    def color_text(self, text, action):
        """Colorize text based on the action."""
        color_codes = {
            "REFLECTION": "\033[94m",  # Blue
            "ACTION": "\033[92m",  # Green
            "OBSERVATION": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "SUBQUESTION": "\033[95m",  # Magenta
            "FINAL ANSWER": "\033[96m",  # Cyan
            "ARGUMENT": "\033[90m",  # Gray
        }
        reset_code = "\033[0m"
        color_code = color_codes.get(action, "")
        return f"{color_code}{text}{reset_code}"

    def save_context_to_html(self, filename="agent_context.html"):
        """Save the agent context to an HTML file."""
        conv = Ansi2HTMLConverter()
        html_content = conv.convert(self.context, full=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Context saved to {filename}")

    # Assistants
    def decompose_question(self, question: str, max_retrials: int = 3) -> dict:
        """Decompose a complex question into simpler parts."""
        decomp_system_prompt = """GENERAL INSTRUCTIONS
        You are an expert in the domain of the following question. Your task is to decompose a complex question into simpler parts.
        
        RESPONSE FORMAT
        {"sub_questions":["<FILL>"]}"""

        for attempt in range(max_retrials):
            answer = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": decomp_system_prompt}, {"role": "user", "content": question}],
            )
            response_content = json.loads(answer.choices[0].message.content)

            try:
                validated_response = DecomposedQuestion.model_validate(response_content)
                return validated_response
            except ValidationError as e:
                print(f"Validation error on attempt {attempt + 1}:", e)

        raise RuntimeError("Maximum number of retries reached without successful validation.")

    def answers_summarizer(self, questions: List[str], answers: List[str], max_retrials: int = 3) -> dict:
        """Summarize a list of answers to the decomposed questions."""
        answer_summarizer_system_prompt = """GENERAL INSTRUCTIONS
        You are an expert in the domain of the following questions. Your task is to summarize the answers to the questions into a single response.
        
        RESPONSE FORMAT
        {"summary": "<FILL>"}"""

        q_and_a_prompt = "\n\n".join(
            [f"SUBQUESTION {i+1}\n{q}\nANSWER {i+1}\n{a}" for i, (q, a) in enumerate(zip(questions, answers))]
        )

        answer = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": answer_summarizer_system_prompt},
                {"role": "user", "content": q_and_a_prompt},
            ],
        )
        response_content = json.loads(answer.choices[0].message.content)
        for attempt in range(max_retrials):
            try:
                validated_response = AnswersSummary.model_validate(response_content)
                return validated_response
            except ValidationError as e:
                print(f"Validation error on attempt {attempt + 1}:", e)

    # Tools
    def math_calculator(self, expression: str) -> float:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)
            return result
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None

    def list_sql_tables(self) -> list:
        """List all tables in the SQL database."""
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            result = self.cursor.fetchall()
            return [table[0] for table in result]
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None

    def sql_db_schema(self, table_name: str) -> str:
        """Return schema of a specific table in the database."""
        try:
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            result = self.cursor.fetchall()
            schema = "\n".join([f"{row[1]} {row[2]}" for row in result])
            return schema
        except Exception as e:
            print(f"Error retrieving schema for table {table_name}: {e}")
            return None

    def sql_db_query(self, query: str) -> str:
        """Run an SQL query and return the result."""
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = openai.OpenAI()

    agent = AgenteReAct(model="gpt-4o-mini", db_path="sql_lite_database.db", memory_path="agent_memory.json")

    question = "How did sales vary between Q1 and Q2 of 2024 in percentage and amount?"
    agent.run_agent(question)
    agent.save_context_to_html("agent_context.html")
    agent.save_memory()
