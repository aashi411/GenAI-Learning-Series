import streamlit as st
import os
import requests

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="City Intelligence",
    page_icon="🌆",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #1e1e1e;
    color: #eeeeee;
}

/* Main title */
.main-title {
    color: #d62828;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #aaaaaa;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #292929;
}

/* Buttons */
.stButton > button {
    background-color: #b91c1c;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #dc2626;
    color: white;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 10px;
}

/* User message */
div[data-testid="stChatMessage"]:has(
    div[data-testid="stChatMessageAvatarUser"]
) {
    background-color: #3a3a3a;
}

/* Assistant message */
div[data-testid="stChatMessage"]:has(
    div[data-testid="stChatMessageAvatarAssistant"]
) {
    background-color: #4a2020;
}

/* Chat input */
div[data-testid="stChatInput"] {
    border-color: #b91c1c;
}

/* Divider */
hr {
    border-color: #444444;
}

/* Info box */
.info-box {
    background-color: #292929;
    border-left: 4px solid #d62828;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TOOLS
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# Tavily
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about the city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n  {snippet[:150]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# ============================================================
# LLM
# ============================================================

llm = ChatMistralAI(
    model="mistral-small-2506"
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = llm.bind_tools(
    [get_weather, get_news]
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None

if "pending_result" not in st.session_state:
    st.session_state.pending_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌆 City Intelligence")

    st.markdown("---")

    st.markdown(
        """
        **Available Tools**

        🌤️ Weather  
        📰 Latest News  

        ---

        **Powered by**

        🤖 Mistral AI  
        🌤️ OpenWeather  
        📰 Tavily
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.session_state.pending_tool = None
        st.session_state.pending_result = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌆 City Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI assistant for weather and latest city news.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    if role == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])

    elif role == "assistant":

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about a city..."
)


if user_input:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # Convert Streamlit messages into LangChain messages
    # --------------------------------------------------------

    messages = []

    for message in st.session_state.messages:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

    # --------------------------------------------------------
    # Ask LLM
    # --------------------------------------------------------

    try:

        with st.spinner("🤖 Thinking..."):

            result = llm_with_tool.invoke(messages)

        # ----------------------------------------------------
        # TOOL REQUIRED
        # ----------------------------------------------------

        if result.tool_calls:

            tool_call = result.tool_calls[0]

            tool_name = tool_call["name"]

            st.session_state.pending_tool = tool_call

            st.session_state.pending_result = result

            # Ask user for permission
            st.rerun()

        # ----------------------------------------------------
        # NORMAL RESPONSE
        # ----------------------------------------------------

        else:

            answer = result.content

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )


# ============================================================
# HUMAN-IN-THE-LOOP TOOL APPROVAL
# ============================================================

if st.session_state.pending_tool:

    tool_call = st.session_state.pending_tool

    tool_name = tool_call["name"]

    st.markdown("---")

    st.markdown(
        f"""
        <div class="info-box">

        🔧 <b>Tool Request</b>

        The AI wants to use:

        <br><br>

        <b>{tool_name}</b>

        <br><br>

        Do you want to allow this tool call?

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    with col1:

        if st.button("✅ Approve"):

            try:

                with st.spinner(
                    f"Using {tool_name}..."
                ):

                    tool_result = tools[
                        tool_name
                    ].invoke(tool_call)

                # Get original AI message
                ai_result = st.session_state.pending_result

                # Create LangChain message list
                messages = []

                for message in st.session_state.messages:

                    if message["role"] == "user":

                        messages.append(
                            HumanMessage(
                                content=message["content"]
                            )
                        )

                    elif message["role"] == "assistant":

                        messages.append(
                            HumanMessage(
                                content=message["content"]
                            )
                        )

                # Add the AI tool request
                messages.append(ai_result)

                # Add tool result
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )

                # Ask Mistral to generate final answer
                final_result = llm_with_tool.invoke(
                    messages
                )

                final_answer = final_result.content

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

                # Reset tool state
                st.session_state.pending_tool = None
                st.session_state.pending_result = None

                st.rerun()

            except Exception as e:

                st.error(
                    f"Tool execution failed: {str(e)}"
                )

    # --------------------------------------------------------
    # DENY
    # --------------------------------------------------------

    with col2:

        if st.button("❌ Deny"):

            denial_message = (
                f"I couldn't use the {tool_name} tool "
                "because you denied the request."
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": denial_message
                }
            )

            st.session_state.pending_tool = None
            st.session_state.pending_result = None

            st.rerun()