import pandas as pd
import json
import openai
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    StringPromptTemplate
)
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI



GPT_MODEL = "gpt-4-0613"
llm = ChatOpenAI(temperature=1.0, model_name=GPT_MODEL)

def business_analysis(mes):
    # Load data from the CSV file
    data = pd.read_csv('output_data.csv')

    data_str = data.to_string()

    # Define a prompt template for business analysis
    business_analysis_template = """
    You are a numeric business analyst. Your goal is to analyze the following business data and provide insights.
    You will find basic measures such as mean, deviation etc but will also spot correlations and give values for the same.
    You will also do in depth detailed analysis of the data and draw enough to make a solid report.
    The data is: 
    {data}
    """

    prompt_for_business_analysis = PromptTemplate(
        input_variables=["data"],
        template=business_analysis_template,
        validate_template=False
    )

    # Initialize the LLMChain with the chat model and the business analysis prompt
    business_analysis_chain = LLMChain(llm=llm, prompt=prompt_for_business_analysis)

    # Get insights from the model
    formatted_prompt = prompt_for_business_analysis.format(data=data_str)
    insight = business_analysis_chain.run(formatted_prompt)

    return(insight)
