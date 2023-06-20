import re
from typing import List, Union
from langchain import OpenAI, LLMChain, SerpAPIWrapper
from langchain.agents import AgentExecutor, AgentOutputParser, LLMSingleActionAgent
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.vectorstores import Pinecone


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": answer}, log=text)
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        return AgentAction(tool=action, tool_input=action_input, log=text)


TEMPLATE = """You are SearchGPT, a professional search engine.
You provide informative answers to users.
Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: if you know the answer, skip to Final Answer, otherwise think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to give detailed, informative answers

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Useful for when you need to answer questions about current events.",
)

embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index("podcasts", embeddings, text_key="text_chunk")
podcasts_qa = RetrievalQA.from_chain_type(OpenAI(), retriever=docsearch.as_retriever())
knowledge_base_tool = Tool(
    name="Knowledge Base",
    func=podcasts_qa.run,
    description=(
        "Useful for general questions about how to do things and for details on "
        "interesting topics. Input should be a fully formed question."
    ),
)

tools = [knowledge_base_tool, search_tool]

prompt = CustomPromptTemplate(
    template=TEMPLATE,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"],
)

agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools],
)
memory = ConversationBufferWindowMemory(k=2)
executor = AgentExecutor.from_agent_and_tools(agent, tools, memory=memory, verbose=True)


def respond(message, history):
    history.append((message, executor.run(message)))
    return "", history
