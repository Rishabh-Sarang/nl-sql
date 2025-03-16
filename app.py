import streamlit as st
import sqlite3
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from typing_extensions import Annotated, TypedDict
import os

# Initialize LLM
llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

def save_csv_to_db(csv_file):
    """Save uploaded CSV to SQLite database."""
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect("data.db")
    df.to_sql("uploaded_data", conn, if_exists="replace", index=False)
    conn.close()

db = SQLDatabase.from_uri("sqlite:///data.db")
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Streamlit UI
st.title("CSV to SQL Query Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    save_csv_to_db(uploaded_file)
    st.success("CSV file uploaded and stored in database!")

user_question = st.text_input("Enter your question")
if st.button("Generate Answer") and user_question:
    state = {"question": user_question}
    state.update(write_query(state))
    state.update(execute_query(state))
    state.update(generate_answer(state))

    
    st.write("### Generated SQL Query")
    st.code(state["query"], language="sql")
    
    st.write("### Query Result")
    st.write(state["result"])
    
    st.write("### Final Answer")
    st.write(state["answer"])
