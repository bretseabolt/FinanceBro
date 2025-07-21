
from langgraph.graph import StateGraph, add_messages, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    messages: Annotated[List, add_messages]


def build_agent_graph(tools: List[BaseTool] = []):

    system_prompt = """
    Your name is Finance Wizard and you are an expert financial advisor. You help users manage spending, build saving habits, and discover beginner-friendly investments.

    <filesystem>
    You have access to a set of tools that allow you to interact with the user's local filesystem.
    You are only allowed to access files within the working directory 'data'.
    The path to this directory is {working_dir}
    </filesystem>

    <data>
    When the user refers to their finances, load/analyze transactions, budgets, etc.
    Use tools for inspection, cleaning, and visualization (e.g., spending distributions).
    </data>

    <financial-tasks>
    - Spending: Categorize transactions (e.g., via ML clustering), track vs. budget.
    - Savings: Suggest habits based on income/expenses (e.g., 50/30/20 rule), set goals.
    - Investments: Use web_search/browse_page for beginner options (e.g., "best index funds for beginners site:investopedia.com"). Always cite sources and warn about risks.
    </financial-tasks>

    <user-interaction>
    Start by asking for financial data upload. When the user first uploads a file, inspect the data with the first 5 rows.
    Summarize insights and suggest actions (e.g., "Your top spending category is food; aim to save 10% there.").
    </user-interaction>

    <tools>
    {tools}
    </tools>

    Assist in a friendly, ethical way. Disclaim: "I'm not a certified advisor; consult professionals."
    """

    llm = ChatGoogleGenerativeAI(name="Finance Wizard", model="gemini-2.5-flash")
    if tools:
        llm = llm.bind(tools=tools)
        tools_json = [tool.model_dump_json(include=['name', 'description']) for tool in tools]

    def assistant(state: AgentState) -> AgentState:
        response = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
        state.messages.append(response)
        return state

    builder = StateGraph(AgentState)

    builder.add_node("Finance_Wizard", assistant)  # Renamed node for theme
    builder.add_node(ToolNode(tools))

    builder.add_edge(START, "Finance_Wizard")
    builder.add_conditional_edges(
        "Finance_Wizard",
        tools_condition,
    )

    builder.add_edge("tools", "Finance_Wizard")

    return builder.compile(checkpointer=MemorySaver())


# visualize graph
if __name__ == "__main__":
    from IPython.display import display, Image

    graph = build_agent_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))
