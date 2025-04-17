import json
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] 
os.environ["LANGSMITH_API_KEY"] 

# Retriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Replace URLs list with file paths and user IDs
file_data = {
    "Student1.txt": "1",
    "Student2_answers.txt": "2",
    "Student2_bio.txt": "2",
    "Student2_exam.txt": "2",
    "Student2_teacher_response.txt": "2",
    "Student3.txt": "3",
}

# Modify loader to use TextLoader and add metadata
docs = []
for file_path, user_id in file_data.items():
    loaded_doc = TextLoader(file_path).load()
    # Add metadata to each document
    for doc in loaded_doc:
        doc.metadata["user_id"] = user_id
    docs.extend(loaded_doc)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100  # Increased chunk size
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
# NO RETRIEVER TOOL NEEDED

# Agent State
from typing import (Annotated, Dict, List, Literal,  # Added Literal and Union
                    Optional, Sequence, TypedDict, Union)

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field  # Import Field
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: Optional[str]  # Add user_id to the state
    docs: Optional[List[Document]] # Store retrieved documents
    max_retries: Annotated[int, Field(default=0)]  # Add retry counter
    route_decision: Optional[Literal["PA", "TA"]] # Router's decision
    grade_decision: Optional[Literal["route", "rewrite", "generate_pa_fallback"]] # Grader's decision


# Nodes and Edges
from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate  # Added line
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition

# NO LONGER NEEDED from langchain.tools.retriever import create_retriever_tool


# --- START: LLM ROUTER DEFINITION ---

class RouteQuery(BaseModel):
   """Route the query to the appropriate agent."""
   route_destination: Literal["PA", "TA"] = Field(
       # UPDATED description based on user feedback
       description="Given the user query, decide whether to route to the Personal Assistant (PA) for concise, factual answers to informational questions, or the Teaching Assistant (TA) for pedagogical support, explanations, and learning-focused guidance."
   )

def route_query(state):
   """
   Routes the query to either the Personal Assistant (PA) or Teaching Assistant (TA).
   """
   print("---ROUTE QUERY---")
   messages = state["messages"]
   question = messages[-1].content # Get the latest user message

   # LLM for routing
   llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash-001")
   structured_llm_router = llm.with_structured_output(RouteQuery)

   # UPDATED Prompt for the router based on user feedback
   system = """You are an expert at routing a user question to the appropriate agent within a Moodle learning environment.
Route to 'PA' (Personal Assistant) if the question is asking for specific, factual information that can likely be answered directly and concisely (e.g., 'What are the notes on topic X?', 'When is the deadline?', 'What was the teacher's feedback on Y?'). The PA uses retrieved context but aims for direct answers.
Route to 'TA' (Teaching Assistant) if the question requires explanation, pedagogical guidance, conceptual understanding, Socratic questioning, or help interpreting feedback to improve learning (e.g., 'Why is X important?', 'Can you explain Y?', 'How can I improve based on this feedback?', 'Help me understand Z.'). The TA uses the Athena pedagogical approach.
"""
   prompt = ChatPromptTemplate.from_messages(
       [
           ("system", system),
           ("human", "{question}"),
       ]
   )

   router_chain = prompt | structured_llm_router
   route_decision = router_chain.invoke({"question": question})

   print(f"---ROUTE DECISION: {route_decision.route_destination}---")

   return {"route_decision": route_decision.route_destination}


def decide_route(state):
   """
   Determines the next node based on the routing decision.
   """
   print(f"---DECIDING ROUTE: {state.get('route_decision')}---")
   route = state.get('route_decision')
   if route == "TA":
       # UPDATED: TA (Teaching Assistant) uses the pedagogical flow starting with retrieve
       print("--- Routing to TA (Pedagogical/Retrieve) ---")
       return "retrieve"
   elif route == "PA":
       # UPDATED: PA (Personal Assistant) goes to a new placeholder node
       print("--- Routing to PA (Informational) ---")
       return "PA_placeholder"
   else:
       # Default or error case, route to TA (pedagogical flow) as a safe default
       print("---WARNING: No route decision found or invalid route, defaulting to TA (Pedagogical/Retrieve)---")
       return "retrieve"

# --- END: LLM ROUTER DEFINITION ---


def retrieve(state):
    """Retrieves documents based on the user's question and ID."""
    print("---RETRIEVE---")
    messages = state["messages"]
    user_message = messages[-1].content  # Assuming the last message is the user's
    user_id = state.get("user_id")  # Get user_id from state

    if not user_id:
        print("Warning: user_id not found in state.")
        return {"docs": []} # Return empty docs if no user_id


    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,  # Number of documents to retrieve
            "filter": {"user_id": user_id}  # Filter by user_id
        }
    )

    retrieved_docs = retriever.invoke(user_message)
    print(f"Retrieved docs: {retrieved_docs}")
    return {"docs": retrieved_docs}

def grade_documents(state: AgentState) -> Dict[str, Literal["route", "rewrite", "generate_pa_fallback"]]:
    """
    Determines whether the retrieved documents are relevant and returns the decision
    in a dictionary to update the state for conditional routing. Adds 'grade_decision' to state.
    """
    print("---GRADE DOCUMENTS---")
    messages = state["messages"]
    question = messages[0].content # Original question
    docs = state.get("docs", [])
    max_retries = state.get("max_retries", 0)
    retry_limit = 1 # Define the retry limit

    # --- Relevance Grading Logic ---
    relevant_docs_found = False
    if docs: # Only grade if docs exist
        print("---GRADING DOCS FOR RELEVANCE---")
        # Data model
        class Grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: bool = Field(description="Relevance score true or false")

        # LLM
        model = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash-001")

        # Prompt
        prompt_template = """You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document:
        {context}
        Here is the user question:
        {question}
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'true' or 'false' to indicate whether the document is relevant to the question."""
        prompt = PromptTemplate.from_template(prompt_template)

        # Chain
        chain = prompt | model | (lambda x: {"binary_score": x.content.lower().strip() == "true"})

        # Join document contents
        doc_content = "\n\n".join(doc.page_content for doc in docs)
        scored_result = chain.invoke({"question": question, "context": doc_content})
        relevant_docs_found = scored_result.get('binary_score', False)

    # --- Decision Logic ---
    decision: Literal["route", "rewrite", "generate_pa_fallback"]
    if relevant_docs_found:
        print("---DECISION: DOCS RELEVANT - ROUTING---")
        decision = "route" # Proceed to router
    else:
        # Docs are not relevant OR no docs were found
        if max_retries < retry_limit:
            print(f"---DECISION: DOCS IRRELEVANT/MISSING - REWRITING (Attempt {max_retries + 1}/{retry_limit})---")
            decision = "rewrite" # Rewrite the query
        else:
            print("---DECISION: DOCS IRRELEVANT/MISSING - MAX RETRIES REACHED - FALLBACK TO PA---")
            decision = "generate_pa_fallback" # Max retries exceeded, go to PA fallback

    # Return the decision in a dictionary to update the state under the key 'grade_decision'
    return {"grade_decision": decision}

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state.
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    # LLM
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash-001")

    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    max_retries = state.get("max_retries", 0)

    # Use a simpler prompt for rewriting
    prompt_template = """
    Look at the input and try to reason about the underlying semantic intent / meaning.

    Here is the initial question:
    {question}

    Formulate an improved question:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Grader
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash-001")
    chain = prompt | model | StrOutputParser() # Use StrOutputParser
    response = chain.invoke({"question": question})
    return {"messages": [HumanMessage(content=response)], "max_retries": max_retries + 1}


def pa_placeholder_node(state):
    """Placeholder node for the Personal Assistant (PA) route."""
    print("---PA PLACEHOLDER NODE---")
    # In a real implementation, this would handle informational queries.
    # For now, it just returns a simple message.
    return {"messages": [HumanMessage(content="[PA Placeholder: This query would be handled by the Personal Assistant.]")]}


def rag_failed_empty_response(state):
    """Node to handle RAG failure by signaling an empty response."""
    print("---RAG FAILED - RETURNING EMPTY SIGNAL---")
    # Use a specific marker string that the Flask endpoint can check
    return {"messages": [HumanMessage(content="RAG_FAILED_EMPTY")]}


def generate_pa_response(state):
    """
    Generate a concise, factual answer using the Personal Assistant persona.
    Retrieves context first if not already present.
    """
    print("---GENERATE PA RESPONSE---")
    messages = state["messages"]
    question = messages[0].content # Use the original question

    # Check if docs are present in the state.
    docs = state.get("docs")
    question = state["messages"][0].content # Get original question

    # --- Define the PA System Prompt FIRST ---
    pa_system_prompt = """**Persona:** You are a Personal Assistant (PA) for a Moodle user. Your goal is to provide concise, factual answers.

    **Core Directive:**
    1.  **Analyze the Query & Context:** Understand the student's question (`{question}`) and review the provided context (`{context}`). The context comes from the user's Moodle documents.
    2.  **Prioritize Context:** If the provided context is relevant and contains the answer, base your response *primarily* on the context. Find the specific information requested.
    3.  **Use General Knowledge as Fallback:** If the provided context is empty, missing, or does not contain the relevant information to answer the question, you MAY use your general knowledge to provide a concise, factual answer.
    4.  **Be Factual & Concise:** Answer the question directly. Do not add opinions or external knowledge beyond what's needed for a direct answer.
    5.  **Reference Context (If Used):** When answering based on the context, implicitly reference the source if possible (e.g., "According to the teacher's feedback...", "The notes mention...").
    6.  **(Disclaimer instruction removed)**
    7.  **Format:** Use clear language. Bullet points can be used for lists if appropriate.

    **Do Not:**
    *   Offer pedagogical advice or explanations (that's the TA's job).
    *   Engage in Socratic dialogue.
    *   Hallucinate information as if it came from the context when it didn't.
    *   Be overly conversational.

    **User Data Context:**
    {context}

    **Student's Question:**
    {question}

    Begin your response by directly addressing the student's question. Follow the rules above regarding context use and the mandatory disclaimer if context is not used.
    """

    # --- Now define LLM, formatting function, and the chain ---
    llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash-001")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(template=pa_system_prompt, input_variables=["context", "question"])
    pa_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    if not docs: # Fallback case: No relevant docs found after retries
        print("---PA: No relevant documents found. Generating response from general knowledge + disclaimer.---")
        # Invoke the chain with empty docs, relying on the updated prompt to handle this
        generation_input = {"docs": [], "question": question}
        response = pa_chain.invoke(generation_input)
        # The LLM (guided by the updated prompt) should add the disclaimer itself.
        return {"messages": [HumanMessage(content=response)]}
    else: # Normal case: Relevant docs found
        print("---PA: Generating response based on retrieved documents.---")
        # Invoke the chain with the retrieved documents
        generation_input = {"docs": docs, "question": question}
        response = pa_chain.invoke(generation_input)
        # LLM should NOT add the disclaimer here (as context is provided)
        return {"messages": [HumanMessage(content=response)]}

    # Note: The original chain definition later in the function is now redundant
    # if we define and use it within the if/else block as shown above.
    # We will remove the later definition.

    # --- If docs ARE present, proceed with normal PA generation ---
    print("---PA: Generating response based on retrieved documents.---")

    # --- The PA System Prompt definition was moved earlier ---
    # --- This block is now empty ---

    # --- The LLM, formatting function, chain definition, and invocation ---
    # --- are now handled within the if/else block near the start of the function ---
    # --- This section below is no longer needed. ---

    # (Removing redundant code previously here)


def generate(state):
    """
    Generate answer using the Athena pedagogical approach.
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content # Use the original question
    docs = state.get("docs", []) # Get retrieved docs.

    # --- NEW ATHENA PROMPT DEFINITION ---
    athena_system_prompt = """**Persona:** You are "Athena," a Moodle AI Teaching Assistant. Your goal is to help students learn effectively.

**Core Directive:**
1.  **Analyze the Query & Context:** Understand the student's question (`{question}`) and review the provided context (`{context}`), which may include relevant notes, past work, or teacher feedback.
2.  **Prioritize Direct Answers:** If the question asks for specific information (e.g., "Are there notes about X?") and the context contains that information, provide it directly and concisely. Reference the source if possible (e.g., "Yes, the teacher's notes cover this...").
3.  **Use Context Intelligently:** Synthesize information from the context to answer the question accurately. If the context reveals understanding or misconceptions, use this to tailor your explanation.
4.  **Adaptive Teaching Method:**
    *   **For informational queries (like asking for notes):** Be direct and concise.
    *   **For conceptual queries ("How does X work?", "Why is Y important?"):** Use a more Socratic approach. Break down concepts, ask guiding questions (CCQs), suggest brief activities, and scaffold learning based on inferred knowledge from the context.
5.  **Conciseness:** Keep explanations focused and avoid unnecessary verbosity. Structure complex information logically (e.g., using bullet points).

**Tone:** Supportive, encouraging, and patient. Facilitate learning, but don't withhold direct answers when appropriate.

**User Data Context:**
{context}

**Student's Question:**
{question}

Begin your response by directly addressing the student's question, using the context where relevant, and applying the appropriate teaching method (direct answer or Socratic guidance).
"""

    # LLM - Using gemini-pro for potentially better reasoning and slightly higher temp
    llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.0-flash-001")

    # Post-processing helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- UPDATED CHAIN with ATHENA PROMPT ---
    # Create a prompt template that uses the system prompt
    prompt = PromptTemplate(template=athena_system_prompt, input_variables=["context", "question"])

    # Chain - Structure to pass context and question correctly to the template
    rag_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run
    generation_input = {"docs": docs, "question": question}
    response = rag_chain.invoke(generation_input)
    # Note: Returning HumanMessage as per original pattern, though AIMessage might be semantically more correct for LLM response.
    return {"messages": [HumanMessage(content=response)]}


print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

# Graph

from langgraph.graph import END, START, StateGraph

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents) # Node that runs the grading logic
workflow.add_node("rewrite", rewrite)
workflow.add_node("router", route_query) # Router node (now called conditionally)
workflow.add_node("generate", generate)  # TA generation
workflow.add_node("generate_pa", generate_pa_response) # PA generation (handles normal and fallback)
workflow.add_node("rag_failed_empty", rag_failed_empty_response) # Node for RAG failure empty response
# Removed: workflow.add_node("agent", agent) - unused
# Removed: workflow.add_node("error", ...) - fallback handled by generate_pa

# --- Define Decision Functions for Conditional Edges ---

def decide_after_grading(state: AgentState) -> Literal["route", "rewrite", "generate_pa_fallback"]:
    """Reads the grade_decision from the state to determine the next step."""
    # grade_documents node adds 'grade_decision' to the state
    decision = state.get("grade_decision")
    print(f"---Deciding after grading: {decision}---")
    if decision not in ["route", "rewrite", "generate_pa_fallback"]:
        print(f"---WARNING: Invalid grade_decision '{decision}' found, defaulting to rewrite---")
        return "rewrite" # Safe default
    return decision

# decide_route function (lines 123-141) is already suitable for the router's conditional edge

# --- Define the Edges ---

# 1. Start with retrieval
workflow.add_edge(START, "retrieve")

# 2. After retrieval, always grade the documents
workflow.add_edge("retrieve", "grade_documents")

# 3. Conditional transitions based on grading outcome
workflow.add_conditional_edges(
    "grade_documents", # Source node
    decide_after_grading, # Function to determine the route based on state['grade_decision']
    {
        "route": "router",             # Relevant docs -> Go to router
        "rewrite": "rewrite",          # Irrelevant/No docs, retries remain -> Rewrite
        "generate_pa_fallback": "rag_failed_empty" # Irrelevant/No docs, max retries -> Go to empty response node
    }
)

# 4. Rewrite loop
workflow.add_edge("rewrite", "retrieve")

# 5. Router decides between TA and PA generation (only if docs were relevant)
workflow.add_conditional_edges(
    "router", # Source node
    decide_route, # Use the existing function based on state['route_decision']
    {
        # Keys match the return values of decide_route
        "retrieve": "generate", # If router decided TA (decide_route returns 'retrieve'), go to TA generate
        "PA_placeholder": "generate_pa", # If router decided PA (decide_route returns 'PA_placeholder'), go to PA generate
    }
)

# 6. Generation nodes lead to END
workflow.add_edge("generate", END)    # TA response generation ends the graph
workflow.add_edge("generate_pa", END) # PA response generation (normal or fallback) ends the graph
workflow.add_edge("rag_failed_empty", END) # RAG failure path ends the graph

# Compile the graph
graph = workflow.compile()

# Inputs
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to receive user messages and return the agent's response.
    """
    try:
        data = request.get_json()
        user_message = data.get("message")
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id") if metadata else None  # Get user ID from metadata

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "user_id missing in metadata"}), 400

        # Add validation for user_id format (positive integer string)
        if not user_id.isdigit() or int(user_id) <= 0:
            # Inform the student about the invalid format
            return jsonify({"error": f"Invalid user ID format: '{user_id}'. Expected a positive whole number."}), 400

        inputs = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id  # Pass user_id in the input
        }

        full_response = ""
        error_message = None # Variable to store potential error message

        for output in graph.stream(inputs):
            # Check which node is producing the output
            for node, value in output.items():
                print(f"--- Node: {node} ---") # Debugging: print current node
                # UPDATED Check: Capture response from EITHER 'generate' (TA) OR 'generate_pa' (PA)
                if node in ["generate", "generate_pa"]:
                    print(f"--- Processing output from final node: {node} ---") # Added debug
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages:
                            last_message = messages[-1]
                            # Robust content extraction
                            message_content = ""
                            if hasattr(last_message, 'content'):
                                message_content = last_message.content
                            elif isinstance(last_message, str): # Handle potential raw strings
                                message_content = last_message

                            if message_content: # Only append if content exists
                                full_response += message_content # Append content
                                print(f"--- Captured response fragment from {node}: '{message_content[:100]}...' ---") # Added debug print (truncated)
                            else:
                                print(f"--- Warning: Message from {node} has no extractable content ---")
                        else:
                             print(f"--- Warning: 'messages' list empty for node {node} ---")
                    else:
                        print(f"--- Warning: Output from {node} is not a dict with 'messages' key ---")

                elif node == "error": # Capture error message
                    print(f"--- Processing output from error node ---") # Added debug
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages:
                            last_message = messages[-1]
                            error_message = last_message.content if not isinstance(last_message, str) else last_message
                            # Optionally break the loop early if error is terminal
                            # break # Uncomment if you want to stop processing immediately on error
            # if error_message: # Uncomment if using the break above
            #     break

        if error_message:
            # Return error if one was captured (e.g., invalid user ID)
            print(f"--- Returning Error Message: {error_message} ---")
            # Use the error message captured if available
            return jsonify({"error": error_message})

        # --- ADD THIS CHECK ---
        elif full_response == "RAG_FAILED_EMPTY":
            # If the specific RAG failure marker is the *only* response content,
            # return a successful response with an empty string.
            print("--- Returning Success: RAG failed, providing empty response for PHP handler. ---")
            return jsonify({"response": ""})
        # --- END ADDED CHECK ---

        elif not full_response:
             # Handle cases where generation might genuinely produce nothing (other than RAG failure)
             print("--- Returning Error: No response generated and no specific error captured. ---")
             return jsonify({"error": "Could not generate a response."}) # Keep this fallback error

        else:
            # Return successful response (from TA or normal PA)
            print(f"--- Returning Success: {full_response} ---")
            return jsonify({"response": full_response}) # Keep normal success return
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/test_error", methods=["GET"])
def test_error():
    """
    Endpoint to test the error handling.
    """
    inputs = {
        "messages": [HumanMessage(content="test error")],
        "user_id": "invalid_id",  # Trigger the error
        "max_retries": 0
    }
    try:
        for output in graph.stream(inputs):
            for node, value in output.items():
                if node == "error":
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages:
                            last_message = messages[-1]
                            return jsonify({"error": last_message.content}), 500
        return jsonify({"error": "An unexpected error occurred."}), 500 # Should not reach here
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# End of ragApp.py
