# Customer Support Agent

A LangGraph-powered customer support agent that combines Retrieval-Augmented Generation (RAG), a PDF knowledge base, local LLM inference using Ollama, tool calling, and automatic escalation detection.

The agent answers customer questions using information retrieved from PDF documents. It can also call Python tools to perform specific tasks, such as checking an order's status and expected delivery date. LangGraph manages the workflow, including conditional routing between the LLM, tools, and the final response.

## Technology Stack

- **Framework:** LangGraph + LangChain
- **LLM:** Ollama — `llama3.2:3b`
- **Embedding Model:** Ollama — `nomic-embed-text`
- **Vector Store:** FAISS
- **Knowledge Base:** PDF documents
- **Tool Execution:** LangGraph `ToolNode`
- **Tool Routing:** `tools_condition`
- **Language:** Python

## Features

- **RAG over PDF Documents:** Loads PDF files, splits the content into chunks, generates embeddings, and retrieves relevant information using FAISS.
- **LLM-Powered Responses:** Uses the local Ollama model to understand customer questions and generate responses.
- **Tool Calling:** Allows the LLM to request the execution of available Python functions.
- **Order Status Lookup:** Checks an order ID and returns its status and expected delivery date using sample order data.
- **LangGraph Workflow:** Manages the conversation through graph nodes, conditional edges, and tool execution.
- **Tool Calling Loop:** Returns tool results to the LLM so it can generate a final response or request another tool.
- **Escalation Detection:** Identifies configured keywords associated with sensitive support issues, such as billing disputes, fraud, and data loss.
- **Conversation History:** Maintains message history during the interactive session.
- **Local Model Execution:** Runs the LLM and embedding model through Ollama without requiring an OpenAI API key.

## Project Workflow

The agent follows a workflow that connects customer messages, the LLM, PDF retrieval, tool execution, and response generation.

1. The customer submits a question.
2. The application stores the message in the LangGraph state.
3. The configured routing and escalation logic is applied.
4. The LLM processes the question and determines whether it needs a tool.
5. For knowledge-based questions, the agent retrieves relevant content from the PDF knowledge base.
6. For order-related questions, the LLM can request the order-status tool.
7. LangGraph routes tool requests to `ToolNode`.
8. The tool executes and returns its result.
9. LangGraph sends the tool result back to the LLM.
10. The LLM generates a customer-friendly response.
11. The graph finishes when no further tool call is requested, or follows the configured escalation route.

## RAG Workflow

The PDF-based RAG pipeline allows the agent to answer questions using the project's knowledge base instead of relying only on the LLM's general knowledge.

1. **Load PDFs:** Read the PDF documents from the configured knowledge base directory.
2. **Split Documents:** Divide the extracted text into smaller chunks.
3. **Generate Embeddings:** Convert document chunks into numerical vectors using `nomic-embed-text`.
4. **Store Vectors:** Store the embeddings in FAISS.
5. **Retrieve Context:** Search for chunks relevant to the customer's question.
6. **Generate Answer:** Provide the retrieved context to the LLM to generate a grounded response.

## Tool Calling Workflow

The project includes an order-status tool that demonstrates how an LLM can request an action from the application.

1. The LLM receives the customer question and the available tool definitions.
2. The LLM decides whether a tool is appropriate and, if so, returns a structured tool call with its arguments.
3. `tools_condition` checks the AI message and routes tool calls to the `tools` node.
4. `ToolNode` executes the requested Python function.
5. The tool result is added to the conversation messages.
6. The graph routes back to the LLM.
7. The LLM uses the tool result to produce a final answer.

The LLM requests the tool call; the Python function performs the actual lookup. Binding a tool to the LLM does not execute the function by itself.

## Order Status Tool

The example `check_order_status` tool accepts an order ID and returns the matching order status and expected delivery date.

Example customer questions:

- Where is my order ORD101?
- What is the status of order ORD102?
- When will my order be delivered?

Example response:

```text
Order ID: ORD101
Status: Shipped
Expected delivery: 18 October 2026
```

The order records are sample data intended for demonstration. They should be replaced with a database or authorized order-management API for real customer support.

## Escalation Detection

The agent includes configurable keyword-based detection for potentially sensitive customer issues, such as billing disputes, fraud, data loss, and serious complaints.

Depending on the implementation, an escalation condition can return a handoff message or route the request to a human-support workflow.

**Note:** Keyword detection alone does not create a real support ticket or contact a human agent. Those actions require a separate ticketing or support-system integration.

## Setup

### 1. Install Python Dependencies

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Install and Prepare Ollama

Install Ollama and download the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Ensure Ollama is running before starting the application.

### 3. Add PDF Knowledge Base Documents

Place your PDF files inside the project's configured knowledge base directory.

Example structure:

```text
13-Customer-Support-Agent/
├── agent.py
├── requirements.txt
├── README.md
└── pdfs/
    ├── product_guide.pdf
    └── support_policy.pdf
```

Use valid, text-readable PDF files. The actual directory name must match the path configured in `agent.py`.

## Run

Start the customer support agent:

```bash
python agent.py
```

If the application supports a configurable knowledge base directory, you can specify it with:

```bash
python agent.py --kb-dir pdfs/
```

The `--kb-dir` option is available only if it is implemented in `agent.py`.

## Example Questions

**Knowledge base questions**
- What does the product guide say about this feature?
- What is the return policy?
- Explain the support policy.

**Tool-based questions**
- Where is my order ORD101?
- Check the status of order ORD102.
- What is the expected delivery date for ORD103?

**Escalation-related questions**
- I want to report fraud.
- I have a serious billing dispute.
- I need help with a data-loss issue.

The actual route depends on the escalation rules, tool definitions, and graph configuration in `agent.py`.

## Configuration

The default model configuration is:

```python
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
```

Update these values if you want to use different compatible Ollama models.

## Troubleshooting

**Ollama model not found**

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**LangGraph tool import error**

Verify that the required packages are installed in the same virtual environment used to run the application:

```powershell
python -m pip list | findstr /i langgraph
```

Test the imports:

```powershell
python -c "from langgraph.prebuilt import ToolNode, tools_condition; print('Tool imports successful')"
```

**No relevant PDF content retrieved**

- Verify that the PDF files exist in the configured directory.
- Check that the files contain readable text.
- Confirm that the document-loading, chunking, and retrieval configuration is correct.

## Future Improvements

- Connect order lookup to a real database or API.
- Add inventory, refund, and cancellation tools.
- Integrate a ticketing system for actual support escalation.
- Add tool execution error handling and logging.
- Add tests for retrieval, tool execution, and graph routing.
- Add authentication and authorization for customer-specific data.

## Disclaimer

This project is intended for learning and demonstration. The sample order data is fictional, and escalation detection does not create a support ticket unless an external support integration is implemented.
