import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain import hub
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key="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"
)

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
    return {"result": execute_query_tool.invoke(state["query"]) }

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

def plot_graph(df):
    columns = df.columns
    
    if len(columns) == 2:
        x_col, y_col = columns
        fig, ax = plt.subplots()
        ax.bar(df[x_col], df[y_col], color='skyblue')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Bar Chart of {y_col} vs {x_col}', fontsize=16)
        ax.set_xticklabels(df[x_col], rotation=45, ha="right", fontsize=12)
    else:
        x_col = columns[1:]
        y_col = columns[0]

        fig, ax = plt.subplots(figsize=(16, 8))

        for col in x_col:
            ax.plot(df[y_col], df[col], marker='o', linestyle='-', label=col)

        ax.set_xlabel(y_col, fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        ax.set_title(f'Line Chart of {", ".join(x_col)} vs {y_col}', fontsize=10)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.tight_layout()
        plt.subplots_adjust(left=0.2, right=0.85, bottom=0.2)
    
    st.pyplot(fig)

query_prompt_template = hub.pull("rishabh/sql-query-system-prompt")

st.title("SQLite Query System with LangChain")

uploaded_file = st.file_uploader("Upload a SQLite (.db) file", type=["db"])

if uploaded_file:
    db_path = f"./{uploaded_file.name}"
    with open(db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    def write_query(question):
        """Generate SQL query to fetch information."""
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": question,
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return result["query"]

    def plot_graph(df):
        """Plot graph based on query result."""
        columns = df.columns
        if len(columns) == 2:
            x_col, y_col = columns
            fig, ax = plt.subplots()
            ax.bar(df[x_col], df[y_col], color='skyblue')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Bar Chart of {y_col} vs {x_col}')
            ax.set_xticklabels(df[x_col], rotation=45, ha="right")
        else:
            x_col = columns[1:]
            y_col = columns[0]
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in x_col:
                ax.plot(df[y_col], df[col], marker='o', linestyle='-', label=col)
            ax.set_xlabel(y_col)
            ax.set_ylabel("Values")
            ax.set_title(f'Line Chart of {", ".join(x_col)} vs {y_col}')
            ax.legend()
        st.pyplot(fig)

    conn = sqlite3.connect(db_path)
    query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query_tables, conn)
    st.write("Tables in Database:", tables)

    question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        try:
            query = write_query(question)
            st.write("Generated SQL Query:")
            st.code(query)

            df = pd.read_sql(query, conn)
            st.write("Query Result:")
            st.write(df)

            plot_graph(df)
        except Exception as e:
            st.error(f"Error: {e}")
    
    conn.close()
