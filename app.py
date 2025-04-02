import os

import streamlit as st

from tools import tools 
from db import db

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.llms import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain.agents import initialize_agent
from langchain.callbacks.base import BaseCallbackHandler

import boto3

# ================= Application =================

load_dotenv()
arango_graph = ArangoGraph(db)
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

llm = Bedrock(client=bedrock_client, model_id="mistral.mistral-large-2402-v1:0")     

def agent_executor(user_query):
    reasoning_steps = []
    
    class CallbackHandler(BaseCallbackHandler):
        def on_agent_action(self, action, **kwargs):
            thought = action.log.split('\n')[0].replace('Thought:', '').strip()
            print(thought)
            step = {
                'type': 'thought',
                'content': f"🤔 **Thought:** {thought}",
                'tool': action.tool,
                'input': action.tool_input
            }
            reasoning_steps.append(step)
            
            with st.sidebar:
                st.markdown(f"**Step {len(reasoning_steps)} - Thought**")
                st.markdown(step['content'])
                st.markdown(f"🔧 **Action:** {step['tool']}")
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

    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True,
            callbacks=[CallbackHandler()],
            handle_parsing_errors=True
        )
        
        # Clear previous steps
        if "reasoning_steps" in st.session_state:
            st.session_state.reasoning_steps = []
            
        result = agent.run(user_query)
        return result, reasoning_steps
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.sidebar.error(error_msg)
        return error_msg, reasoning_steps

if "messages" not in st.session_state:
    st.session_state.messages = []

# st.sidebar.title("Reasoning Steps")
# st.sidebar.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your drug-related query..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.sidebar.empty()
    st.sidebar.title("Reasoning Steps")
    st.sidebar.divider()

    with st.spinner("Thinking..."):
        result, _ = agent_executor(user_input)

    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
