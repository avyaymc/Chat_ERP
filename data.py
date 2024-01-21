
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
from business_analysis import business_analysis
#from conn_gpt import query_database, get_db_schema
from graph_gt import open_graph

GPT_MODEL = "gpt-3.5-turbo-0613"
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
        description="The function to use second when asked to perform analysis. This function takes a single string input which is the filename. Use it to find useful outputs for user to understand their data. ALWAYS do your own analysis too." 
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

def datarelated(query):
    answer=dbagent.run(query)
    return answer
