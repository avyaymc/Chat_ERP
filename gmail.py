from langchain.agents.agent_toolkits import GmailToolkit
import os
toolkit = GmailToolkit()

from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()

from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)


def mailrelated(query):
    answer = agent.run(query)
    return answer
