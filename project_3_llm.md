# Project 3: LLM Code Review Agent

## 🎯 Goal
Create an autonomous agent that can execute external Python code/tools (like a linter) based on natural language instructions and then summarize the findings.

## 🛠️ Technology Stack
* **Orchestration:** LangChain Agents and Tools
* **LLM:** Any LLM API (e.g., OpenAI, recommended for strong reasoning)
* **Tools:** Standard Python libraries (e.g., `pylint`, file I/O).

## ⚙️ Steps & Milestones

### 1. Define Custom Tools
1.  **Code Reading Tool:** Create a Python function (e.g., `read_code(file_path: str) -> str`) that securely reads a specified file. Use LangChain's **`tool`** decorator or `Tool` class to wrap this function.
2.  **Linter Tool:** Create a function (e.g., `run_linter(code: str) -> str`) that executes an external utility like `pylint` on the provided code string and returns the output. Wrap this as a second Tool.

### 2. Initialize the Agent
1.  **LLM:** Load your chosen LLM (e.g., `ChatOpenAI`).
2.  **Agent Prompt:** Write a clear **system prompt** that defines the Agent's personality, goal (to review code for best practices), and constraints (it must use its tools before answering).
3.  **Initialization:** Use `initialize_agent` (or the newer LCEL equivalent) by providing the list of custom **Tools** and the **LLM**.

### 3. Execution and Observation
1.  **Run:** Pass an instruction to the Agent: "Please review the code in `test_script.py` and tell me what could be improved."
2.  **Trace:** Observe the Agent's internal thought process:
    * **Thought:** I need to read the file first, so I will call the `read_code` tool.
    * **Action:** `read_code(file_path='test_script.py')`
    * **Observation:** (The returned code content)
    * **Thought:** Now I should check for technical debt, so I will call the `run_linter` tool.
    * **Action:** `run_linter(...)`
    * **Final Answer:** Generate the review based on the tool outputs.

### 💡 Key Learning Points
* The difference between a **Chain** (fixed steps) and an **Agent** (dynamic reasoning).
* How to equip an LLM with **external functionality** using the **`Tool`** concept.
* Analyzing the **Agent's thought process** to understand why it chooses specific actions.
