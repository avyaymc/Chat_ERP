import mysql.connector
import pandas as pd
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-4-0613"

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + "",
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# MySQL connection URI
uri = 'mysql+mysqlconnector://root:password@localhost:3306/althea'

config = {
    'user': 'root',
    'password': 'Test123$',
    'host': 'localhost',
    'database': 'altheadb',
    'port': '3306',
}

# Establish a connection to the MySQL server
connection = mysql.connector.connect(**config)



def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
            )
        )



def get_table_names(conn):
    """Return a list of table names."""
    cursor=conn.cursor()
    table_names = []
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'altheadb';")
    for table in cursor.fetchall():
        table_names.append(table[0])
    cursor.close()
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    cursor=conn.cursor()
    column_names = []
    cursor.execute(f"SELECT `COLUMN_NAME` FROM `INFORMATION_SCHEMA`.`COLUMNS` WHERE `TABLE_SCHEMA`='altheadb' AND `TABLE_NAME`='{table_name}';")
    columns = cursor.fetchall()
    for col in columns:
        column_names.append(col[0])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts


def get_db_schema():
    database_schema_dict = get_database_info(connection)
    database_schema_string = "\n".join(
        [
            f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
            for table in database_schema_dict
        ]
    )
    return database_schema_string

database_schema_string=get_db_schema()

functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about stock inventory. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            The query should follow the MySQL syntax format.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    cursor=conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except Exception as e:
        results = f"query failed with error: {e}"
    cursor.close
    return results

def execute_function_call(message):
    if message["function_call"]["name"] == "ask_database":
        query = json.loads(message["function_call"]["arguments"])["query"]
        results = ask_database(connection, query)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results

def query_database(mes):
    connection = mysql.connector.connect(**config)
    messages = []
    messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the MySQL Inventory Database."})
    messages.append({"role": "user", "content": mes})
    chat_response = chat_completion_request(messages, functions)
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    if assistant_message.get("function_call"):
        results = execute_function_call(assistant_message)
        messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})
    pretty_print_conversation(messages)
    connection.close()
    return results

