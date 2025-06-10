import os
import streamlit as st
import json
import base64
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
from typing import List, Dict, Any, Optional
import hashlib

from tools import tools

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# ================= Memory & History Management =================

class ChatHistory(BaseChatMessageHistory):
    """SQLite-based chat message history with metadata and search capabilities."""
    
    def __init__(self, session_id: str, database_path: str = "chat_history.db"):
        self.session_id = session_id
        self.database_path = database_path
        self._create_tables()
    
    def _create_tables(self):
        """Create the messages table if it doesn't exist."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0,
                    tool_calls TEXT,
                    reasoning_step TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_time 
                ON messages(session_id, timestamp)
            """)
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages for the current session."""
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (self.session_id,)
            )
            
            messages = []
            for row in cursor:
                if row['message_type'] == 'human':
                    messages.append(HumanMessage(
                        content=row['content'],
                        additional_kwargs={
                            'timestamp': row['timestamp'],
                            'metadata': json.loads(row['metadata'] or '{}'),
                            'id': row['id']
                        }
                    ))
                elif row['message_type'] == 'ai':
                    messages.append(AIMessage(
                        content=row['content'],
                        additional_kwargs={
                            'timestamp': row['timestamp'],
                            'metadata': json.loads(row['metadata'] or '{}'),
                            'tool_calls': json.loads(row['tool_calls'] or '[]'),
                            'reasoning_step': row['reasoning_step'],
                            'id': row['id']
                        }
                    ))
            return messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        message_type = 'human' if isinstance(message, HumanMessage) else 'ai'
        metadata = json.dumps(message.additional_kwargs.get('metadata', {}))
        tool_calls = json.dumps(message.additional_kwargs.get('tool_calls', []))
        reasoning_step = message.additional_kwargs.get('reasoning_step', '')
        token_count = self._count_tokens(message.content)
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO messages 
                (session_id, message_type, content, metadata, token_count, tool_calls, reasoning_step)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id, message_type, message.content, 
                metadata, token_count, tool_calls, reasoning_step
            ))
    
    def clear(self) -> None:
        """Clear all messages for the current session."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
    
    def _count_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def get_recent(self, limit: int = 10) -> List[BaseMessage]:
        """Get the most recent N messages."""
        messages = self.messages
        return messages[-limit:] if len(messages) > limit else messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    SUM(token_count) as total_tokens,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM messages WHERE session_id = ?
            """, (self.session_id,))
            
            row = cursor.fetchone()
            return {
                'total_messages': row[0],
                'total_tokens': row[1],
                'first_message': row[2],
                'last_message': row[3],
                'session_id': self.session_id
            }
    
    def search(self, query: str, limit: int = 5) -> List[BaseMessage]:
        """Search messages by content."""
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE session_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.session_id, f"%{query}%", limit))
            
            messages = []
            for row in cursor:
                if row['message_type'] == 'human':
                    messages.append(HumanMessage(content=row['content']))
                else:
                    messages.append(AIMessage(content=row['content']))
            return messages


class ConversationMemory:
    """Manages conversation memory with context window optimization."""
    
    def __init__(self, session_id: str, max_tokens: int = 8000):
        self.session_id = session_id
        self.max_tokens = max_tokens
        self.chat_history = ChatHistory(session_id)
    
    def get_context(self) -> List[BaseMessage]:
        """Get conversation history optimized for context window."""
        messages = self.chat_history.messages
        
        if not messages:
            return []
        
        # Calculate total tokens
        total_tokens = sum(len(msg.content) // 4 for msg in messages)
        
        if total_tokens <= self.max_tokens:
            return messages
        
        # If too many tokens, use a sliding window approach
        # Keep the first message (context) and recent messages
        if len(messages) > 2:
            first_message = messages[0]
            recent_messages = []
            current_tokens = len(first_message.content) // 4
            
            # Add recent messages until we hit the token limit
            for msg in reversed(messages[1:]):
                msg_tokens = len(msg.content) // 4
                if current_tokens + msg_tokens > self.max_tokens:
                    break
                recent_messages.insert(0, msg)
                current_tokens += msg_tokens
            
            return [first_message] + recent_messages
        
        return messages[-1:]  # Just keep the last message if only 2 messages
    
    def save_interaction(self, user_input: str, ai_response: str, 
                       tool_calls: List = None, reasoning_step: str = ""):
        """Add a complete interaction to memory."""
        # Add user message
        human_msg = HumanMessage(
            content=user_input,
            additional_kwargs={
                'timestamp': datetime.now().isoformat(),
                'metadata': {'session_id': self.session_id}
            }
        )
        self.chat_history.add_message(human_msg)
        
        # Add AI response
        ai_msg = AIMessage(
            content=ai_response,
            additional_kwargs={
                'timestamp': datetime.now().isoformat(),
                'metadata': {'session_id': self.session_id},
                'tool_calls': tool_calls or [],
                'reasoning_step': reasoning_step
            }
        )
        self.chat_history.add_message(ai_msg)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        stats = self.chat_history.get_stats()
        recent_topics = self._find_topics()
        
        return {
            **stats,
            'recent_topics': recent_topics,
            'context_status': self._get_context_status()
        }
    
    def _find_topics(self, limit: int = 3) -> List[str]:
        """Extract recent conversation topics using keyword extraction."""
        recent_messages = self.chat_history.get_recent(6)
        topics = []
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                # Simple keyword extraction - in production, use NLP libraries
                words = msg.content.lower().split()
                important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                if important_words:
                    topics.extend(important_words[:2])
        
        return list(set(topics))[:limit]
    
    def _get_context_status(self) -> str:
        """Get the status of context window usage."""
        messages = self.chat_history.messages
        total_tokens = sum(len(msg.content) // 4 for msg in messages)
        
        usage_percent = (total_tokens / self.max_tokens) * 100
        
        if usage_percent < 50:
            return "optimal"
        elif usage_percent < 80:
            return "good"
        elif usage_percent < 95:
            return "near_limit"
        else:
            return "exceeds_limit"


def create_session_id() -> str:
    """Generate or retrieve session ID."""
    if 'session_id' not in st.session_state:
        # Create a unique session ID based on timestamp and random component
        timestamp = str(int(datetime.now().timestamp()))
        random_component = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        st.session_state.session_id = f"session_{timestamp}_{random_component}"
    
    return st.session_state.session_id


def load_chat_history(session_id: str) -> ChatHistory:
    """Get message history for a session."""
    return ChatHistory(session_id)


# ================= Callback Handler with Memory =================

class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, conversation_memory: ConversationMemory):
        super().__init__()
        self.reasoning_steps = []
        self.conversation_memory = conversation_memory
        self.tool_calls = []

    def on_agent_action(self, action, **kwargs):
        # Extract meaningful reasoning from the action context
        if hasattr(action, 'log') and action.log:
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
            'input': action.tool_input,
            'timestamp': datetime.now().isoformat()
        }
        self.reasoning_steps.append(step)
        self.tool_calls.append({
            'tool': action.tool,
            'input': action.tool_input,
            'timestamp': datetime.now().isoformat()
        })
        
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
                'content': f"✅ {final_answer}",
                'timestamp': datetime.now().isoformat()
            }
            with st.sidebar:
                st.markdown(f"**Final Answer**")
                st.success(step['content'])
                st.divider()


# ================= Agent Executor with Memory =================

def setup_agent(conversation_memory: ConversationMemory):
    """Create an agent executor with memory integration."""
    
    # Get optimized conversation history
    chat_history = conversation_memory.get_context()
    
    # Create prompt with memory integration
    system_prompt = """You are an AI assistant specialized in drug discovery and pharmaceutical research. You can ONLY use the tools that are explicitly provided to you.

CRITICAL RULES:
- You can ONLY use the tools listed in your available tools - no web search, no internet access, no external databases
- If you don't have a tool to get specific information, clearly state this limitation
- Do NOT pretend to search online or access external resources
- Base your responses only on the tool results you receive
- If a user asks for information you cannot obtain with available tools, explain what tools you would need
- Be honest about your limitations when tools are missing

TOOL SELECTION STRATEGY:
- Use specific tools (FindDrug, FindProteinsFromDrug, etc.) when you need exact, focused information
- Consider TextToAQL when you need to explore relationships, complex queries, or when specific tools don't cover the user's question
- TextToAQL can handle broad biomedical questions and database relationships that other tools might not address
- Choose the most appropriate tool based on the specific information requested

CONVERSATION CONTEXT:
You have access to our conversation history. Use this context to provide more personalized and relevant responses. Reference previous interactions when appropriate.

When you need to use a tool, explain your reasoning clearly and use the appropriate tool for the task."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    return prompt, chat_history


def agent_executor(user_query: str, conversation_memory: ConversationMemory):
    """Execute agent with memory integration."""
    try:
        prompt, chat_history = setup_agent(conversation_memory)
        
        # Create agent with memory-aware prompt
        agent = create_tool_calling_agent(llm, tools, prompt)
        callback_handler = CustomCallbackHandler(conversation_memory)

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
        
        # Clear previous reasoning steps
        if "reasoning_steps" in st.session_state:
            st.session_state.reasoning_steps = []

        # Execute with chat history
        final_state = agent_executor.invoke({
            "input": user_query,
            "chat_history": chat_history
        })

        output_text = final_state["output"].strip()
        
        # Store the interaction in memory
        reasoning_summary = " | ".join([step.get('content', '') for step in callback_handler.reasoning_steps])
        conversation_memory.save_interaction(
            user_input=user_query,
            ai_response=output_text,
            tool_calls=callback_handler.tool_calls,
            reasoning_step=reasoning_summary
        )
        
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return output_text
        
    except Exception as e:
        print(f"Agent execution error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        st.sidebar.error(error_msg)
        return error_msg


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
    model="gemini-2.5-flash-preview-05-20", 
    google_api_key=google_api_key, 
    temperature=0,
    convert_system_message_to_human=True
)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("logo.png")

# ================= Initialize Memory =================

session_id = create_session_id()
conversation_memory = ConversationMemory(session_id)

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

# Add clear memory button to top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear Memory", type="secondary", help="Clear conversation history"):
        conversation_memory.chat_history.clear()
        st.session_state.messages = []
        st.rerun()

st.chat_message("assistant").markdown(
    "👋 **Welcome to NeuThera!**\n\n"
    "You're currently using the **MVP** version of the app, which only includes a limited set of drug discovery tools. (Mostly due to limited resources)\n\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load existing conversation from memory on first run
if not st.session_state.messages and conversation_memory.chat_history.messages:
    for msg in conversation_memory.chat_history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

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
        result = agent_executor(user_input, conversation_memory)
    
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})