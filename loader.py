import os
from llama_hub.file.base import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
import openai
from datetime import datetime, timedelta
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from langchain.agents import AgentExecutor, initialize_agent, Tool, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain, MapReduceChain, create_tagging_chain, summarize
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    StringPromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
#from conn_gpt import query_database, get_db_schema
from graph_gt import open_graph
from business_analysis import business_analysis
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chains import PALChain
import json
import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

GPT_MODEL = "gpt-3.5-turbo-0613"

loader = SimpleDirectoryReader('./data/documents', recursive=True, exclude_hidden=True)
documents = loader.load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

llm_normal = ChatOpenAI(temperature=0, model_name=GPT_MODEL)


normal_chat_template = """
You are a personal assistant for AltheaSuite, a cloud-based ERP. Your goal is to assist the user in completing their tasks. 
Please respond with a nice tone and maintain friendly conversation.
If the user asks for their past conversation history, respond based only on the history from {conversation_history}. 
The user input is {input}. 
"""

prompt_for_normal_chat = PromptTemplate(
    input_variables=["input", "conversation_history"],
    template=normal_chat_template,
    validate_template=False
)
normal_chat_chain = LLMChain(llm=llm_normal, prompt=prompt_for_normal_chat)




llm = ChatOpenAI(temperature=0, model_name=GPT_MODEL)

#schema=get_db_schema()



def save_data_for_graph(data):
    filename='graph_data.csv'  # Change the file extension to .csv
    data = data.split('\n')  # Split the data into a list of strings
    with open(filename, 'w') as f:
        # Assume the first row of the data contains the column names
        column_names = data[0].split(',')
        f.write(f"{column_names[0]},{column_names[1]}\n")  # Write the header

        # Write the data points
        for row in data[1:]:
            x, y = row.split(',')
            f.write(f"{x},{y}\n")  # Use a comma instead of a tab

    return f"Data saved to {filename} for graph generation."

def save_data_for_analysis(data):
    filename='output_data.csv'  # Change the file extension to .csv
    data = data.split('\n')  # Split the data into a list of strings
    with open(filename, 'w') as f:
        # Assume the first row of the data contains the column names
        column_names = data[0].split(',')
        f.write(','.join(column_names) + "\n")  # Write the header

        # Write the data points
        for row in data[1:]:
            values = row.split(',')
            f.write(','.join(values) + "\n")  # Use a comma instead of a tab

    return f"Data saved to {filename} for analysis."

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

toolsdb = [
    Tool(
        name="OpenGraph",
        func=open_graph,
        description="The function to use second when you have to plot a graph. Use it after calling the \"SaveDataForGraph\" function. Your job is to figure out what the graph is supposed to be. Send a string saying either /'pie/', /'hist/' or '/box/', depending on whether the user asked for a pie graph, histogram graph or box plot graph respectively. If you're not able to figure out what kind of graph the user wants, ask them to clarify between the three graphs mentioned. You do not generate the graph, the function does, inform the user of the same",
    ),
    # Tool(
    #     name="QueryDatabase",
    #     func=query_database,
    #     description=f"This function queries the MySQL database. Use the schema ({schema})",
    # ),
    Tool(
        name="SaveDataForGraph",
        func=save_data_for_graph,
        description="The function to use first when asked to plot a graph. This function takes a string, where each line represents a data point in 'x,y' format. The first line should contain the column names. The function saves the data to a CSV file, which can then be used to generate a graph. If the input is already a string dont make an outer string.",
    ),
    Tool(
        name="SaveDataForAnalysis",
        func=save_data_for_analysis,
        description="The function to use first when asked to perform analysis. This function takes a string, where each line represents a data point. The first line should contain the column names. The function saves the data to a CSV file, which can then be used for analysis. This function can handle any number of columns. If the input is already a string dont make an outer string.",
    ),
    Tool(
        name="BusinessAnalysis",
        func=business_analysis,
        description="The function to use second when asked to perform analysis. This function takes a single string which is the filename. Use it to find useful outputs for user to understand their data. ALWAYS do your own analysis too." 
    ),
    Tool.from_function(
        name="Calculator",
        func=llm_math_chain.run,
        description="Use whenever you need to perform math operations. This function takes a string with the calculation to be performed worded in natural language.",
    )
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

dbagent = initialize_agent(toolsdb, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,agent_kwargs=agent_kwargs, memory=memory)



    
def format_query(query_type, input_text, conversation_history):
    conversation_history_str = " ".join(conversation_history)
    return {'conversation_history': conversation_history, 'input': prompt_for_normal_chat.format(input=input_text, conversation_history=conversation_history_str)}

tagging_schema_main = {
    "properties": {
        "mode": {
            "type": "string", 
            "enum": ["normal chat", "documents", "data", "tasks"],# "graph"],
            "description": "Determines the mode of operation. 'normal chat' for regular conversation, 'documents' for document retrieval, 'data' for data based operations and 'tasks' for discussing tasks"
        }
    }
}
llm_normal = ChatOpenAI(temperature=0, model_name=GPT_MODEL)
tagging_chain = create_tagging_chain(tagging_schema_main, llm_normal)


def save_history():
    summary = summarize_history()
    with open('past_history.txt', 'w') as f:
        f.write(summary)

def summarize_history():
    llm = OpenAI(temperature=0)
    text_splitter = CharacterTextSplitter()
    with open("past_history.txt") as f:
        past_history = f.read()
    texts = text_splitter.split_text(past_history) + conversation_history
    docs = [Document(page_content=t) for t in texts[:3]]
    summary_template = """Write a summary of the input. 
    Ensure it doesnt contain summaries about the conversation but rather about the exact numbers discussed or the exact task discussed in one or two lines.
    The input is: {text}. """
    PROMPT_summary = PromptTemplate(template=summary_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT_summary)
    summary = chain.run(docs)
    return summary


class Task:
    def __init__(self, task_id, description, roles, assigned_to):
        self.task_id = task_id
        self.description = description
        self.roles = roles
        self.assigned_to = assigned_to

    def complete(self):
        self.is_completed = True

class Employee:
    def __init__(self, name, role, employees):
        self.name = name
        self.role = role
        self.employees = employees


def load_tasks_from_json(filename, employee_name, role, employees):
    with open(filename) as f:
        tasks_data = json.load(f)
    tasks = {}
    for task_data in tasks_data:
        assigned_to = task_data.get('assigned_to', '')
        task_role = task_data.get('roles', [])

        # If the employee is the one the task is assigned to
        if employee_name == assigned_to:
            task = Task(task_id=task_data.get('id', ''), description=task_data.get('description', ''), roles=task_role, assigned_to=assigned_to)
            tasks[task.task_id] = task

        # If the employee is a manager and the task is assigned to one of their subordinates
        elif role == "manager" and assigned_to in [e['name'] for e in employees if e['manager'] == employee_name]:
            task = Task(task_id=task_data.get('id', ''), description=task_data.get('description', ''), roles=task_role, assigned_to=assigned_to)
            tasks[task.task_id] = task

        # If the employee is an executive, they have access to their own tasks and all managers' tasks
        elif role == "executive" and assigned_to != employee_name and 'executive' not in task_role:
            task = Task(task_id=task_data.get('id', ''), description=task_data.get('description', ''), roles=task_role, assigned_to=assigned_to)
            tasks[task.task_id] = task

    return tasks




def load_employee_role_from_json(filename, employee_name):
    with open(filename) as f:
        employees_data = json.load(f)
    for employee_data in employees_data:
        if 'name' in employee_data and employee_name == employee_data['name']:
            return employee_data.get('role', ''), employees_data
    return None, None


def add_task(task_description: str, assigned_to=None, roles=None):
    if assigned_to is None:
        assigned_to = employee.name
    if roles is None:
        roles = employee.role
    tasks = load_tasks_from_json('tasks.json', employee.name, employee.role, employee.employees)
    # Compute the new task_id
    if tasks:
        max_task_id = max(int(task_id) for task_id in tasks.keys())
        task_id = str(max_task_id + 1)
    else:
        task_id = "1"
    new_task = {"id": task_id, "description": task_description, "assigned_to": assigned_to, "roles": [roles]}
    tasks_data = []
    with open('tasks.json', 'r') as f:
        tasks_data = json.load(f)
    tasks_data.append(new_task)
    with open('tasks.json', 'w') as f:
        json.dump(tasks_data, f)
    return f"Task with id {task_id} added successfully."



def delete_task(task_description: str):
    tasks_data = []
    with open('tasks.json', 'r') as f:
        tasks_data = json.load(f)
    updated_tasks_data = [task for task in tasks_data if task['description'] != task_description]
    if len(tasks_data) == len(updated_tasks_data):
        return f"No task with the description '{task_description}' found."
    else:
        with open('tasks.json', 'w') as f:
            json.dump(updated_tasks_data, f)
        return f"Task with description '{task_description}' deleted successfully."


def list_tasks(str):
    tasks = load_tasks_from_json('tasks.json', employee.name, employee.role, employee.employees)
    tasks_str = "\n".join([f"Task ID: {task_id}, Description: {task.description}, Assigned to: {task.assigned_to}" for task_id, task in tasks.items()])
    return tasks_str


toolstask = [
    Tool(
        name="AddTask",
        func=add_task,
        description="Call this function ONLY after calling ListTasks to understand task structure. This function adds a new task to the task management system. Normally it takes only a string input of task_description. If assigned to and roles are mentioned in query then it takes a string input which should be formatted as 'task_description,assigned_to,roles'. Each field is separated by a comma."
    ),
    Tool(
        name="DeleteTask",
        func=delete_task,
        description="Call this function ONLY after calling ListTasks to look at the task descriptions. This function deletes a task from the task management system. It takes a string input which should be the EXACT MATCHING task description of the task to be deleted."
    ),
    Tool(
        name="ListTasks",
        func=list_tasks,
        description="This function retrieves a list of all tasks from the task management system. It takes a single string as input which is the exact query made by the user. DO NOT FORGET TO SEND THE INPUT"
    )
]


agent_kwargs_tasks = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory_tasks = ConversationBufferMemory(memory_key="memory", return_messages=True)

taskagent = initialize_agent(toolstask, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,agent_kwargs=agent_kwargs_tasks, memory=memory_tasks)

current_mode = "chat"
conversation_history = []



def get_answer(query, employee):
    # Extract employee details
    employee_name = employee.name
    employee_role = employee.role
    employees = employee.employees
    global current_mode
    global conversation_history
    
    
    conversation_history.append(f"User: {query}")
    answer = ""

    if query.lower() == "exit":
        current_mode = "chat"
        answer = f"Switching to mode: {current_mode}"
    else:
        if current_mode == "chat":
            tagged_input = tagging_chain.run(query)
            query_type = tagged_input.get('mode', current_mode)
            if query_type in ["documents", "data", "tasks"]:
                current_mode = query_type
                answer = f"Switching to mode: {current_mode}"
                if current_mode == "documents":
                    answer += "\n" + query_engine.query(query)
                elif current_mode == "data":
                    answer += "\n" + dbagent.run(query) 
                elif current_mode == "tasks":
                    answer += "\n" + taskagent.run(query)
            else:
                formatted_prompt = format_query(query_type, query, conversation_history)
                answer = normal_chat_chain.run(formatted_prompt)
        elif current_mode == "documents":
            answer = query_engine.query(query)
        elif current_mode == "data":
            answer = dbagent.run(query) 
        elif current_mode == "tasks":
            answer = taskagent.run(query)

    conversation_history.append(f"Bot: {answer}")
    save_history()
    return answer

employee_name = input("Please enter your name:\n> ")
employee_role, employees = load_employee_role_from_json('employees.json', employee_name)
employee = Employee(employee_name, employee_role, employees)


while True:
    query = input(">")
    answer = get_answer(query, employee)
    print(answer)