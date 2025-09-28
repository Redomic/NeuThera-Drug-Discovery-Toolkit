import os
import streamlit as st
import json
import base64
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
from typing import List, Dict, Any, Optional
import hashlib

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Import the enhanced multi-agent system
from multi_agent import EnhancedReWOOOrchestrator

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
                # Simple keyword extraction
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


# ================= Enhanced Multi-Agent Executor =================

def enhanced_multiagent_executor(user_query: str, conversation_memory: ConversationMemory):
    """Execute enhanced multi-agent pipeline with proper memory integration."""
    
    # Get conversation history for context
    conversation_history = conversation_memory.get_context()
    
    # Update orchestrator with conversation history
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = EnhancedReWOOOrchestrator(
            conversation_history=conversation_history
        )
    else:
        # Update existing orchestrator with new conversation history
        st.session_state.orchestrator.update_conversation_history(conversation_history)
    
    orchestrator = st.session_state.orchestrator
    
    # Process the query
    exec_result = orchestrator.process_query(user_query)

    # Display execution details in the sidebar
    with st.sidebar:
        st.markdown("### 🎯 Enhanced Execution Plan")
        
        if exec_result.plan.steps:
            for step in exec_result.plan.steps:
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌", 
                    "pending": "⏳"
                }.get(step.status, "❓")
                
                st.markdown(f"**{status_emoji} Step {step.step_id}**: {step.tool_name}")
                st.markdown(f"📥 **Input**: `{step.tool_input}`")
                
                if step.dependencies:
                    st.markdown(f"🔗 **Dependencies**: {step.dependencies}")
                
                if step.status == "failed" and step.error:
                    st.error(f"Error: {step.error}")
                
                if step.execution_time:
                    st.markdown(f"⏱️ **Time**: {step.execution_time:.2f}s")
                
                st.divider()

        st.markdown("### 🛠️ Tool Outputs")
        for step_id, output in exec_result.tool_outputs.items():
            if output is not None:
                # Find corresponding step
                step_name = next((s.tool_name for s in exec_result.plan.steps if s.step_id == step_id), f"Step {step_id}")
                
                pretty_output = str(output)
                if len(pretty_output) > 400:
                    pretty_output = pretty_output[:400] + " …"
                st.markdown(f"**{step_name}**: `{pretty_output}`")
            else:
                st.markdown(f"**Step {step_id}**: ❌ No output")
            st.divider()

        # Show execution summary
        st.markdown("### 📊 Execution Summary")
        st.markdown(f"**Success**: {'✅' if exec_result.success else '❌'}")
        st.markdown(f"**Total Time**: {exec_result.execution_time:.2f}s")
        st.markdown(f"**Tools Used**: {len(exec_result.plan.steps)}")
        
        if not exec_result.success and exec_result.error:
            st.error(f"Error: {exec_result.error}")

    # Save the interaction in memory with detailed tool information
    tool_calls_info = [
        {
            "step": step.step_id,
            "tool": step.tool_name,
            "status": step.status,
            "execution_time": step.execution_time
        }
        for step in exec_result.plan.steps
    ]
    
    conversation_memory.save_interaction(
        user_input=user_query,
        ai_response=exec_result.final_answer,
        tool_calls=tool_calls_info,
        reasoning_step=" → ".join([s.tool_name for s in exec_result.plan.steps])
    )

    return exec_result.final_answer


# ================= Application Setup =================

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

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

img_base64 = get_base64_image("logo.png")

# ================= Initialize Memory =================

session_id = create_session_id()
conversation_memory = ConversationMemory(session_id)

# ================= Streamlit UI =================

# Add clear memory button to top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear Memory", type="secondary", help="Clear conversation history"):
        conversation_memory.chat_history.clear()
        st.session_state.messages = []
        if 'orchestrator' in st.session_state:
            del st.session_state.orchestrator
        st.rerun()

# Welcome message
st.chat_message("assistant").markdown(
    "👋 **Welcome to NeuThera Enhanced!**\n\n"
    "You're now using the **Enhanced Multi-Agent** version with improved reasoning, better error handling, and context-aware planning.\n\n"
    "✨ **New Features:**\n"
    "- Context-aware tool selection based on conversation history\n"
    "- Robust error handling and fallback strategies\n"
    "- Enhanced result synthesis\n"
    "- Better AQL query generation\n\n"
    "Ask me anything about drug discovery, molecular analysis, or pharmaceutical research!"
)

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load existing conversation from memory on first run
if not st.session_state.messages and conversation_memory.chat_history.messages:
    for msg in conversation_memory.chat_history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Type your drug-related query..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Setup sidebar
    st.sidebar.empty()
    if img_base64:
        st.sidebar.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{img_base64}' width='175'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h1 style='text-align: center; color: #F3BB4F; font-size: 2rem;'>Enhanced Research Agent</h1>", unsafe_allow_html=True)
    st.sidebar.divider()

    # Process query with enhanced multi-agent system
    with st.spinner("🧠 Processing with enhanced multi-agent system..."):
        try:
            result = enhanced_multiagent_executor(user_input, conversation_memory)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            result = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})