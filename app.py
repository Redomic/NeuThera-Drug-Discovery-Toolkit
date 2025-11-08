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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Import the enhanced drug discovery orchestrator
from drug_orchestration import DrugDiscoveryOrchestrator

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
    """Execute the Drug Discovery Orchestrator with full workflow and memory support."""

    # Retrieve conversation history for context
    conversation_history = conversation_memory.get_context()

    # Initialize orchestrator (store in session for persistence)
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = DrugDiscoveryOrchestrator(
            conversation_history=conversation_history
        )
    else:
        st.session_state.orchestrator.update_conversation_history(conversation_history)

    orchestrator = st.session_state.orchestrator

    # Run the workflow-aware orchestrator
    exec_result = orchestrator.process_query(user_query)

    # --- Sidebar Summary ---
    with st.sidebar:
        st.markdown("### 🧬 Workflow Analysis")
        
        # Determine workflow type display
        if exec_result.plan.is_drug_discovery_workflow:
            workflow_type_display = "Multi-Stage Discovery"
        else:
            workflow_type_display = "Direct Query"
            
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <p style='color: white; margin: 0; font-size: 0.9em;'><strong>Workflow Type:</strong> {workflow_type_display}</p>
            <p style='color: white; margin: 5px 0 0 0; font-size: 0.9em;'><strong>Execution Time:</strong> {exec_result.execution_time:.2f}s</p>
            <p style='color: white; margin: 5px 0 0 0; font-size: 0.9em;'><strong>Total Steps:</strong> {len(exec_result.plan.steps)}</p>
            <p style='color: white; margin: 5px 0 0 0; font-size: 0.9em;'><strong>Status:</strong> {'✅ Success' if exec_result.success else '❌ Failed'}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Execution Pipeline")
        for i, step in enumerate(exec_result.plan.steps, 1):
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "pending": "⏳"
            }.get(step.status, "❔")

            # Use workflow_stage instead of stage.value
            stage_display = step.workflow_stage if step.workflow_stage else "N/A"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                <p style='color: white; margin: 0; font-weight: bold;'>{status_emoji} Step {i}: {step.tool_name}</p>
                <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 0.85em;'>Stage: {stage_display}</p>
                <p style='color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.8em;'>Input: {step.tool_input[:50]}...</p>
                {f"<p style='color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.8em;'>⏱️ {step.execution_time:.2f}s</p>" if step.execution_time else ""}
            </div>
            """, unsafe_allow_html=True)

        # Use workflow_summary instead of workflow_report
        if exec_result.workflow_summary:
            st.markdown("### 📊 Workflow Summary")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 15px; border-radius: 10px;'>
                <p style='color: white; margin: 0; font-size: 0.9em;'>{exec_result.workflow_summary}</p>
            </div>
            """, unsafe_allow_html=True)

    # Save this interaction
    tool_calls_info = [
        {
            "step": step.step_id,
            "tool": step.tool_name,
            "status": step.status,
            "execution_time": step.execution_time,
            "stage": step.workflow_stage if step.workflow_stage else "N/A"
        }
        for step in exec_result.plan.steps
    ]

    conversation_memory.save_interaction(
        user_input=user_query,
        ai_response=exec_result.final_answer,
        tool_calls=tool_calls_info,
        reasoning_step=" → ".join([s.tool_name for s in exec_result.plan.steps])
    )

    # Return the orchestrator's final synthesized response
    return exec_result.final_answer

     



# ================= Application Setup =================

hide_streamlit_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with gradient background */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* User message with gradient */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px;
        padding: 16px 20px;
        border: none;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }

    /* Assistant message with gradient */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:first-child {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border-radius: 18px;
        padding: 16px 20px;
        border: none;
        box-shadow: 0 8px 24px rgba(240, 147, 251, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Avatar styling */
    [data-testid="stChatMessageAvatarUser"], 
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 50%;
        width: 40px;
        height: 40px;
    }
    
    /* Style chat input with glowing effect */
    .stChatInput > div {
        border-radius: 30px;
        border: 2px solid transparent;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-clip: padding-box;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stChatInput input {
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.05);
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Welcome card styling */
    .welcome-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    /* Feature badges */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9em;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        [data-testid="stChatMessage"] > div:first-child {
            padding: 12px 16px;
            font-size: 14px;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Text color for better visibility */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Markdown content in messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: white !important;
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
    if st.button("🗑️ Clear", type="secondary", help="Clear conversation history"):
        conversation_memory.chat_history.clear()
        st.session_state.messages = []
        if 'orchestrator' in st.session_state:
            del st.session_state.orchestrator
        st.rerun()

# Enhanced welcome message using chat message
with st.chat_message("assistant"):
    st.markdown("### 💊 **Welcome to NeuThera Enhanced!**")
    st.markdown("**Next-Generation Multi-Agent Drug Discovery Platform**")
    st.markdown("")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🧠 **Adaptive AI**")
    with col2:
        st.markdown("⚡ **Lightning Fast**")
    with col3:
        st.markdown("🔬 **Research-Grade**")
    with col4:
        st.markdown("🎯 **High Precision**")
    
    st.markdown("")
    st.markdown("#### ✨ Advanced Capabilities:")
    st.markdown("""
    - **🧬 Intelligent Workflow Detection:** Automatically identifies multi-stage drug discovery processes
    - **⚙️ Smart Error Recovery:** Advanced fallback strategies ensure continuous operation
    - **🔍 Enhanced Result Synthesis:** Clear, actionable insights from complex data
    - **💡 Optimized AQL Queries:** Faster database analysis with intelligent query generation
    """)
    
    st.markdown("")
    st.markdown("*Powered by advanced multi-agent orchestration | Real-time molecular analysis | Pharmaceutical intelligence*")


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
if user_input := st.chat_input("🔬 Ask about drug discovery, molecular research, or pharmaceutical insights..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Setup sidebar
    st.sidebar.empty()
    if img_base64:
        st.sidebar.markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='data:image/png;base64,{img_base64}' width='150' style='border-radius: 15px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 1.8rem; margin-bottom: 20px;'>
        Research Agent
    </h1>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    # Process query with enhanced multi-agent system
    with st.spinner("🧬 Analyzing query with multi-agent orchestration..."):
        try:
            result = enhanced_multiagent_executor(user_input, conversation_memory)
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            result = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})