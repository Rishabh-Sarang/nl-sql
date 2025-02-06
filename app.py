import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain import hub
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict, Annotated
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3, os

load_dotenv()  # Load environment variables, if needed

# Initialize LLM and prompt template
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key="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") #Use Streamlit secrets for API key
query_prompt_template = hub.pull("rishabh/sql-query-system-prompt")
assert len(query_prompt_template.messages) == 1

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

        fig, ax = plt.subplots(figsize=(16, 8))  # Increase figure size for better spacing

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

# Streamlit UI
st.title("SQL Query Interface")

uploaded_file = st.file_uploader("Choose a .db file", type="db")

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with open("uploaded.db", "wb") as f:
            f.write(uploaded_file.getbuffer())

        db_path = "uploaded.db"  # Use the temporary file path
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        conn = sqlite3.connect(db_path)
        st.success(f"Connected to database: {uploaded_file.name}")

        question = st.text_area("Enter your SQL query or question:", height=150)

        if st.button("Execute"):
            if question:
                try:
                    with st.spinner("Generating and executing query..."):
                        state = {"question": question}

                        query_output = write_query(state)
                        state["query"] = query_output["query"]
                        st.write("**Generated Query:**")
                        st.code(state["query"], language="sql")

                        result_output = execute_query(state)
                        state["result"] = result_output["result"]

                        df = pd.read_sql(state["query"], conn)
                        df = df.loc[:, ~df.columns.duplicated()]
                        st.write("**Dataframe:**")
                        st.dataframe(df)

                        if not df.empty:
                            plot_graph(df)

                        answer_output = generate_answer(state)
                        state["answer"] = answer_output["answer"]
                        st.write("**Answer:**")
                        st.write(state["answer"])

                except Exception as e:
                    st.error(f"An error occurred: {e}")

            else:
                st.warning("Please enter a question or query.")

    except Exception as e:
        st.error(f"Error processing the database file: {e}")
    finally:
        # Clean up the temporary file (optional, but good practice)
        if os.path.exists(db_path):
            os.remove(db_path)


