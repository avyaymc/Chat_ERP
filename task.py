import os
import re
import json
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from typing import Dict, List, Any, Union, Callable
from pydantic import BaseModel, Field
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts.base import StringPromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish
from langchain import LLMMathChain
from langchain.agents import AgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, initialize_agent, Tool, AgentType




GPT_MODEL = "gpt-3.5-turbo-0613"
#schema=get_db_schema()
llm = ChatOpenAI(temperature=0, model_name=GPT_MODEL)

class Task:
    def __init__(self, task_id, description, roles, assigned_to):
        self.task_id = task_id
        self.description = description
        self.roles = roles
        self.assigned_to = assigned_to

    def complete(self):
        self.is_completed = True


def load_tasks_from_json(filename, employee_name):
    with open(filename) as f:
        tasks_data = json.load(f)
    tasks = {}
    for task_data in tasks_data:
        assigned_to = task_data.get('assigned_to', '')

        # If the employee is the one the task is assigned to
        if employee_name == assigned_to:
            task = Task(task_id=task_data.get('id', ''), description=task_data.get('description', ''), roles=task_data.get('roles', []), assigned_to=assigned_to)
            tasks[task.task_id] = task

    return tasks


def add_task(employee_name, task_description: str):
    tasks = load_tasks_from_json('tasks.json', employee_name)
    # Compute the new task_id
    if tasks:
        max_task_id = max(int(task_id) for task_id in tasks.keys())
        task_id = str(max_task_id + 1)
    else:
        task_id = "1"
    new_task = {"id": task_id, "description": task_description, "assigned_to": employee_name, "roles": []}
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


def list_tasks(employee_name):
    tasks = load_tasks_from_json('tasks.json', employee_name)
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
        description="This function retrieves a list of all tasks from the task management system. It takes a single string as input which is the exact query made by the user. DO NOT FORGET TO SEND THE INPUT. List the EXACT OUTPUT WITHOUT MAKING CHANGES."
    )
]


agent_kwargs_tasks = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory_tasks = ConversationBufferMemory(memory_key="memory", return_messages=True)

taskagent = initialize_agent(toolstask, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,agent_kwargs=agent_kwargs_tasks, memory=memory_tasks)

def taskrelated(query):
    answer = taskagent.run(query)
    return answer