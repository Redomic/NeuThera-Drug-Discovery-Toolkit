import os
import streamlit as st
import json
import base64
from dotenv import load_dotenv

from tools import tools

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler



# ================= Application =================

hide_streamlit_style = """
    <style>
    /* Hide Streamlit header, footer, and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide "Deploy" button */
    .stDeployButton {display: none;}
    
    /* Remove padding and margins for full embed */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Remove sidebar completely for embedded view */
    .css-1d391kg {display: none;}
    
    /* Adjust chat message styling for embedding */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:first-child {
        background: linear-gradient(90deg, #F3BB4F 0%, #E8A935 100%) !important;
        color: white !important;
        border-radius: 12px;
        padding: 12px 16px;
        border: none;
        box-shadow: 0 2px 8px rgba(243, 187, 79, 0.2);
    }

    /* Assistant message styling */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:first-child {
        background: linear-gradient(90deg, #16ADA9 0%, #128A87 100%) !important;
        color: white !important;
        border-radius: 12px;
        padding: 12px 16px;
        border: none;
        box-shadow: 0 2px 8px rgba(22, 173, 169, 0.2);
    }
    
    /* Style chat input */
    .stChatInput > div {
        border-radius: 25px;
        border: 2px solid #16ADA9;
    }
    
    .stChatInput input {
        border-radius: 25px;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Responsive design for mobile embedding */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        [data-testid="stChatMessage"] > div:first-child {
            padding: 8px 12px;
            font-size: 14px;
        }
    }

    
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=google_api_key, 
    temperature=0,
    convert_system_message_to_human=True
)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("logo.png")

class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.reasoning_steps = []

    def on_agent_action(self, action, **kwargs):
        # For tool calling agents, the thought process is handled differently
        # Extract meaningful reasoning from the action context
        if hasattr(action, 'log') and action.log:
            # Parse the reasoning from the log
            thought_content = action.log.strip()
        else:
            # Create contextual thoughts based on the tool being used
            if action.tool == "FindDrug":
                thought_content = f"I need to search for detailed information about the drug: {action.tool_input}"
            elif action.tool == "DrugInteractions":
                thought_content = f"I should check for potential drug interactions with: {action.tool_input}"
            elif action.tool == "MolecularInfo":
                thought_content = f"Let me get molecular information for: {action.tool_input}"
            elif action.tool == "PlotSmiles3D":
                thought_content = f"I'll generate a 3D structure from the SMILES: {action.tool_input}"
            else:
                thought_content = f"I'll use the {action.tool} tool to help answer this question"

        if action.tool == "_Exception":
            raise ValueError("Agent attempted an invalid action: _Exception") 
        
        step = {
            'type': 'thought',
            'content': f"🤔 **Thought:** {thought_content}",
            'tool': action.tool,
            'input': action.tool_input
        }
        self.reasoning_steps.append(step)
        
        with st.sidebar:
            st.markdown(f"**Step {len(self.reasoning_steps)} - Reasoning**")
            st.markdown(step['content'])
            st.markdown(f"🔧 **Tool:** {step['tool']}")
            st.markdown(f"📤 **Input:** `{step['input']}`")
            st.divider()
    
    def on_agent_finish(self, finish, **kwargs):
        if finish.log:
            final_answer = finish.log
            step = {
                'type': 'answer',
                'content': f"✅ {final_answer}"
            }
            with st.sidebar:
                st.markdown(f"**Final Answer**")
                st.success(step['content'])
                st.divider()

def agent_executor(user_query):
    try:
        # Use ChatPromptTemplate with MessagesPlaceholder for better tool calling support
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in drug discovery and pharmaceutical research. You can ONLY use the tools that are explicitly provided to you.
            
            CRITICAL RULES:
            - You can ONLY use the tools listed in your available tools - no web search, no internet access, no external databases
            - If you don't have a tool to get specific information, clearly state this limitation
            - Do NOT pretend to search online or access external resources
            - Base your responses only on the tool results you receive
            - If a user asks for information you cannot obtain with available tools, explain what tools you would need
            - Be honest about your limitations when tools are missing
            
            When you need to use a tool, explain your reasoning clearly and use the appropriate tool for the task."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use create_tool_calling_agent instead of create_react_agent for better Gemini compatibility
        agent = create_tool_calling_agent(llm, tools, prompt)
        callback_handler = CustomCallbackHandler()

        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            callbacks=[callback_handler], 
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=8,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )
        
        if "reasoning_steps" in st.session_state:
            st.session_state.reasoning_steps = []

        final_state = agent_executor.invoke({"input": user_query})

        print("DEBUG: Final State Output →", final_state)

        output_text = final_state["output"].strip()
        
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return output_text
        
    except Exception as e:
        print(f"Agent execution error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        st.sidebar.error(error_msg)
        return error_msg


# Streamlit UI Logic
st.markdown(
    """
    <style>
    /* Change user message to bronze */
    [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) > div:first-child {
        background-color: #F3BB4F !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px;
    }

    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:first-child {
        background-color: #16ADA9 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px;
    }
    """,
    unsafe_allow_html=True
)

st.chat_message("assistant").markdown(
    "👋 **Welcome to NeuThera!**\n\n"
    "You're currently using the **MVP** version of the app, which only includes a limited set of drug discovery tools. (Mostly due to limited resources)\n\n"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your drug-related query..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.sidebar.empty()
    st.sidebar.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{img_base64}' width='175'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h1 style='text-align: center; color: #F3BB4F; font-size: 2rem;'>Research Agent</h1>", unsafe_allow_html=True)
    st.sidebar.divider()

    with st.spinner("Thinking..."):
        result = agent_executor(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})