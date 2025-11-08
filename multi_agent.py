import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import hashlib
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

# Import your existing tools
from tools import (
    FindDrug, FindProteinsFromDrug, TextToAQL, 
    PlotSmiles2D, PlotSmiles3D, PredictBindingAffinity,
    GetAminoAcidSequence, GetChemBERTaEmbeddings,
    PreparePDBData, GenerateCompounds, FindSimilarDrugs,
    AnalyzeProtein, PredictADMETProperties,
    PredictDisorderRegionsinProteins, AnalyzeProteinConservation,
    PredictDrugDrugInteractions, PredictLigandBindingSites, EnumerateTautomersAndStereoisomers, GenerateContactMapFromPDB , PredictCYP450Sites,
    PredictBloodBrainBarrierPenetration,PredictVeberRules,AutoExtractQSARFeatures,PredictSyntheticAccessibility
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Data Classes ==================

@dataclass
class ToolCall:
    """Represents a single tool call in the execution plan."""
    step_id: int
    tool_name: str
    tool_input: Any
    dependencies: List[int] = field(default_factory=list)
    output: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class ExecutionPlan:
    """Represents the complete execution plan for a query."""
    query: str
    steps: List[ToolCall]
    final_synthesis_prompt: str
    created_at: datetime = field(default_factory=datetime.now)
    execution_metadata: Dict = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Contains the final result of plan execution."""
    query: str
    plan: ExecutionPlan
    final_answer: str
    execution_time: float
    tool_outputs: Dict[int, Any]
    success: bool
    error: Optional[str] = None

# ================== Enhanced Tool Registry ==================

class EnhancedToolRegistry:
    """
    Enhanced tool registry that properly handles tool execution with context preservation.
    """
    
    def __init__(self, conversation_history: List[BaseMessage] = None):
        self.tools = self._initialize_tools()
        self.tool_descriptions = self._generate_descriptions()
        self.conversation_history = conversation_history or []
        
    def _initialize_tools(self) -> Dict[str, callable]:
        """Initialize the tool mapping with proper error handling."""
        return {
            "FindDrug": FindDrug,
            "FindProteinsFromDrug": FindProteinsFromDrug,
            "TextToAQL": TextToAQL,
            "PlotSmiles2D": PlotSmiles2D,
            "PlotSmiles3D": PlotSmiles3D,
            "PredictBindingAffinity": PredictBindingAffinity,
            "GetAminoAcidSequence": GetAminoAcidSequence,
            "GetChemBERTaEmbeddings": GetChemBERTaEmbeddings,
            "PreparePDBData": PreparePDBData,
            "GenerateCompounds": GenerateCompounds,
            "FindSimilarDrugs": FindSimilarDrugs,
            "AnalyzeProtein": AnalyzeProtein,
            "PredictADMETProperties": PredictADMETProperties,
            "PredictDisorderRegionsinProteins": PredictDisorderRegionsinProteins,
            "AnalyzeProteinConservation": AnalyzeProteinConservation,
            "PredictDrugDrugInteractions": PredictDrugDrugInteractions,
            "PredictLigandBindingSites": PredictLigandBindingSites,
            "EnumerateTautomersAndStereoisomers": EnumerateTautomersAndStereoisomers,
            "GenerateContactMapFromPDB": GenerateContactMapFromPDB,
            "PredictCYP450Sites": PredictCYP450Sites,
            "PredictBloodBrainBarrierPenetration": PredictBloodBrainBarrierPenetration,
            "PredictVeberRules": PredictVeberRules,
            "AutoExtractQSARFeatures": AutoExtractQSARFeatures,
            "PredictSyntheticAccessibility": PredictSyntheticAccessibility
        }
    
    def _generate_descriptions(self) -> str:
        """Generate detailed tool descriptions for better planning."""
        descriptions = []
        tool_details = {
            "FindDrug": "Search for detailed drug information by name, returning SMILES, molecular data, and properties",
            "FindProteinsFromDrug": "Find proteins that interact with a specific drug compound",
            "TextToAQL": "Execute complex biomedical database queries using natural language - use for broad research questions",
            "PlotSmiles2D": "Generate 2D molecular structure visualizations from SMILES strings",
            "PlotSmiles3D": "Generate 3D molecular structure visualizations from SMILES strings", 
            "PredictBindingAffinity": "Predict binding affinity between drug compounds and protein targets",
            "GetAminoAcidSequence": "Retrieve amino acid sequences for specific proteins",
            "GetChemBERTaEmbeddings": "Generate molecular embeddings for chemical compounds",
            "PreparePDBData": "Prepare protein structure data from PDB for analysis",
            "GenerateCompounds": "Generate new chemical compounds with desired properties",
            "FindSimilarDrugs": "Find structurally similar drugs to a given compound",
            "AnalyzeProtein": "Perform comprehensive protein analysis including structure and function",
            "PredictADMETProperties": "Predict Absorption, Distribution, Metabolism, Excretion, and Toxicity properties",
            "PredictDisorderRegionsinProteins": "Identify disordered regions in protein structures",
            "AnalyzeProteinConservation": "Analyze evolutionary conservation patterns in proteins",
            "PredictDrugDrugInteractions": "Predict potential interactions between multiple drugs",
            "PredictLigandBindingSites": "Identify potential binding sites for ligands on proteins",
            "EnumerateTautomersAndStereoisomers": "Generate all possible tautomers and stereoisomers for a given compound",
            "GenerateContactMapFromPDB": "Create contact maps from protein structures in PDB files",
            "PredictCYP450Sites": "Predict sites of metabolism by CYP450 enzymes on drug compounds",
            "PredictBloodBrainBarrierPenetration": "Predict if a compound can penetrate the blood-brain barrier",
            "PredictVeberRules": "Evaluate drug-likeness based on Veber's rules",
            "AutoExtractQSARFeatures": "Automatically extract QSAR features from chemical structures",
            "PredictSyntheticAccessibility": "Predict the synthetic accessibility of chemical compounds"
        }
        
        for name, func in self.tools.items():
            description = tool_details.get(name, "Tool for drug discovery research")
            descriptions.append(f"- {name}: {description}")
        
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, tool_input: Any) -> Any:
        """Execute a tool with proper error handling and context preservation."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in registry")
        
        tool_func = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        try:
            # Handle the input properly - some tools expect strings, others expect structured data
            if isinstance(tool_input, dict):
                result = tool_func(**tool_input)
            else:
                result = tool_func(tool_input)
                
            logger.info(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
            logger.error(error_msg)
            # Return a structured error instead of raising, so the pipeline can continue
            return {
                "error": True,
                "message": error_msg,
                "tool": tool_name,
                "input": tool_input
            }

# ================== Context-Aware Planner ==================

SYSTEM_PROMPT = """You are an AI assistant specialized in drug discovery and pharmaceutical research. You can ONLY use the tools that are explicitly provided to you.

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

class ContextAwarePlannerAgent:
    """
    Enhanced planner that considers conversation history and tool capabilities.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI, tool_registry: EnhancedToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_planning_prompt()
    
    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create an enhanced planning prompt with better context."""
        return ChatPromptTemplate.from_template("""
You are an expert drug discovery research planner. Create an efficient execution plan WITHOUT seeing tool outputs.

CONVERSATION HISTORY:
{conversation_history}

AVAILABLE TOOLS:
{tool_descriptions}

USER QUERY: {query}

PLANNING GUIDELINES:
1. Use TextToAQL for broad biomedical research questions and database exploration
2. Use specific tools (FindDrug, PredictBindingAffinity, etc.) for targeted data retrieval
3. Consider the conversation history - avoid repeating recent searches
4. Plan for error handling - have fallback strategies
5. Keep tool inputs simple and direct

Generate a JSON plan:
{{
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "ToolName",
            "tool_input": "simple_input_value",
            "dependencies": [],
            "description": "What this accomplishes",
            "reasoning": "Why this tool was chosen"
        }}
    ],
    "synthesis_prompt": "How to combine results into a comprehensive answer"
}}

IMPORTANT RULES:
- Keep tool_input values simple (strings, not complex objects)
- Use dependencies (#step_id) only when absolutely necessary
- Prefer single-step solutions when possible
- Choose the most appropriate tool for each task

Examples:

For "Tell me about aspirin":
{{
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "FindDrug",
            "tool_input": "aspirin",
            "dependencies": [],
            "description": "Get comprehensive aspirin information",
            "reasoning": "FindDrug provides detailed drug data including SMILES and properties"
        }}
    ],
    "synthesis_prompt": "Present the aspirin information in a structured format with key properties highlighted"
}}

For "What proteins interact with aspirin?":
{{
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "TextToAQL", 
            "tool_input": "proteins that interact with aspirin",
            "dependencies": [],
            "description": "Search for aspirin-protein interactions",
            "reasoning": "TextToAQL can handle broad protein interaction queries"
        }}
    ],
    "synthesis_prompt": "List the proteins that interact with aspirin and explain their significance"
}}

Generate the plan:
""")
    
    def create_plan(self, query: str) -> ExecutionPlan:
        """Create an execution plan with conversation context."""
        logger.info(f"Creating contextual execution plan for: {query}")
        
        # Format conversation history
        history_text = ""
        if self.tool_registry.conversation_history:
            recent_messages = self.tool_registry.conversation_history[-4:]  # Last 4 messages
            for msg in recent_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content[:200]}...\n"
        
        response = self.llm.invoke(
            self.prompt.format(
                conversation_history=history_text or "No previous conversation",
                tool_descriptions=self.tool_registry.tool_descriptions,
                query=query
            )
        )
        
        try:
            # Clean the response to extract JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            plan_data = json.loads(content)
            
            # Convert to ToolCall objects
            steps = []
            for step_data in plan_data["steps"]:
                tool_call = ToolCall(
                    step_id=step_data["step_id"],
                    tool_name=step_data["tool_name"],
                    tool_input=step_data["tool_input"],
                    dependencies=step_data.get("dependencies", [])
                )
                steps.append(tool_call)
            
            plan = ExecutionPlan(
                query=query,
                steps=steps,
                final_synthesis_prompt=plan_data.get("synthesis_prompt", "Provide a comprehensive answer")
            )
            
            logger.info(f"Created plan with {len(steps)} steps")
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse planner response: {e}")
            logger.error(f"Raw response: {response.content}")
            # Create a simple fallback plan
            return self._create_smart_fallback_plan(query)
    
    def _create_smart_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create an intelligent fallback plan based on query keywords."""
        query_lower = query.lower()
        
        # Simple keyword-based tool selection
        if any(word in query_lower for word in ["find", "search", "tell me about", "what is"]):
            if any(drug in query_lower for drug in ["drug", "medication", "compound"]):
                tool_name = "FindDrug"
                # Extract potential drug name
                words = query.split()
                drug_candidates = [w for w in words if len(w) > 3 and w.isalpha()]
                tool_input = drug_candidates[-1] if drug_candidates else query
            else:
                tool_name = "TextToAQL"
                tool_input = query
        else:
            tool_name = "TextToAQL"
            tool_input = query
        
        return ExecutionPlan(
            query=query,
            steps=[
                ToolCall(
                    step_id=1,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    dependencies=[]
                )
            ],
            final_synthesis_prompt="Provide a clear and comprehensive answer to the user's question"
        )

# ================== Robust Worker Agent ==================

class RobustWorkerAgent:
    """Enhanced worker agent with better error handling and context preservation."""
    
    def __init__(self, tool_registry: EnhancedToolRegistry):
        self.tool_registry = tool_registry
    
    def execute_step(self, step: ToolCall, context: Dict[int, Any]) -> Tuple[int, Any]:
        """Execute a step with robust error handling."""
        logger.info(f"Executing step {step.step_id}: {step.tool_name}")
        
        try:
            # Resolve input dependencies if any
            resolved_input = self._resolve_input(step.tool_input, context)
            
            # Execute the tool
            start_time = datetime.now()
            result = self.tool_registry.execute_tool(step.tool_name, resolved_input)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check if result contains an error
            if isinstance(result, dict) and result.get("error"):
                step.status = "failed"
                step.error = result["message"]
                logger.error(f"Step {step.step_id} failed: {result['message']}")
                return (step.step_id, None)
            
            step.output = result
            step.status = "completed"
            step.execution_time = execution_time
            
            logger.info(f"Step {step.step_id} completed successfully in {execution_time:.2f}s")
            return (step.step_id, result)
            
        except Exception as e:
            error_msg = f"Step {step.step_id} failed with exception: {str(e)}"
            logger.error(error_msg)
            step.status = "failed"
            step.error = str(e)
            return (step.step_id, None)
    
    def _resolve_input(self, input_spec: Any, context: Dict[int, Any]) -> Any:
        """Resolve input dependencies with better error handling."""
        if not isinstance(input_spec, str) or not input_spec.startswith("#"):
            return input_spec
        
        try:
            # Parse step reference
            if "." in input_spec:
                # Handle complex references like "#1.smiles"
                parts = input_spec[1:].split(".")
                step_id = int(parts[0])
                
                if step_id not in context or context[step_id] is None:
                    logger.warning(f"Referenced step {step_id} not found or failed")
                    return input_spec  # Return original if can't resolve
                
                result = context[step_id]
                for part in parts[1:]:
                    if isinstance(result, dict):
                        result = result.get(part)
                    else:
                        result = getattr(result, part, None)
                
                return result if result is not None else input_spec
            else:
                # Simple step reference like "#1"
                step_id = int(input_spec[1:])
                return context.get(step_id, input_spec)
                
        except Exception as e:
            logger.warning(f"Failed to resolve input {input_spec}: {str(e)}")
            return input_spec

# ================== Enhanced Solver Agent ==================

class EnhancedSolverAgent:
    """Enhanced solver with better synthesis and error handling."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_solver_prompt()
    
    def _create_solver_prompt(self) -> ChatPromptTemplate:
        """Create an enhanced solver prompt."""
        return ChatPromptTemplate.from_template(f"""
{SYSTEM_PROMPT}

SYNTHESIS TASK:
You are synthesizing results from LOCAL TOOLS ONLY. You do NOT have web access.

ORIGINAL QUERY: {{query}}

TOOL EXECUTION RESULTS (FROM LOCAL TOOLS ONLY):
{{tool_outputs}}

SYNTHESIS INSTRUCTIONS: {{synthesis_prompt}}

Based on the LOCAL tool results above, provide a comprehensive and accurate response:

CRITICAL SYNTHESIS GUIDELINES:
1. ONLY use information from the local tool results provided
2. If tools failed or returned limited data, acknowledge these limitations honestly
3. Do NOT pretend to have access to web search or external databases
4. If the query requires information not available through local tools, clearly explain this
5. Be transparent about the source of all information (local database/tools)
6. Suggest what additional tools would be needed for missing information

RESPONSE STRUCTURE:
- Focus on answering the original query directly
- Highlight key scientific findings from LOCAL sources
- If any tools failed, acknowledge limitations
- Provide context for complex scientific concepts
- Be honest about any missing information due to tool limitations
- Suggest follow-up questions that CAN be answered with available local tools

IMPORTANT:
- Only use information from the tool results provided
- Be transparent that all data comes from local sources, not web searches
- If results are incomplete, explain what local information is missing
- Format molecular structures, sequences, and data clearly

Response:
""")
    
    def synthesize(self, query: str, tool_outputs: Dict[int, Any], 
                   synthesis_prompt: str, execution_plan: ExecutionPlan) -> str:
        """Synthesize results with enhanced error handling."""
        logger.info("Synthesizing final answer with system prompt constraints")
        
        # Format outputs with error handling
        formatted_outputs = self._format_outputs_enhanced(tool_outputs, execution_plan)
        
        try:
            response = self.llm.invoke(
                self.prompt.format(
                    query=query,
                    tool_outputs=formatted_outputs,
                    synthesis_prompt=synthesis_prompt
                )
            )
            return response.content
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return f"""I apologize, but I encountered an error while synthesizing the results from our local tools: {str(e)}

Based on the available local tool outputs:
{formatted_outputs}

Please note: I can only use the local tools available to me and do not have web access or external database connectivity. If you need additional information, please let me know what specific aspects you'd like me to explore with our available local tools."""
    
    def _format_outputs_enhanced(self, outputs: Dict[int, Any], plan: ExecutionPlan) -> str:
        """Enhanced output formatting with step context."""
        formatted = []
        
        for step in plan.steps:
            step_id = step.step_id
            output = outputs.get(step_id)
            
            formatted.append(f"Step {step_id}: {step.tool_name}")
            formatted.append(f"  Input: {step.tool_input}")
            formatted.append(f"  Status: {step.status}")
            
            if step.status == "completed" and output is not None:
                output_str = str(output)
                # Truncate very long outputs but preserve structure
                if len(output_str) > 1000:
                    output_str = output_str[:1000] + "... [truncated for brevity]"
                formatted.append(f"  Result: {output_str}")
            elif step.status == "failed":
                formatted.append(f"  Error: {step.error or 'Unknown error'}")
            else:
                formatted.append(f"  Result: No output available")
            
            formatted.append("")  # Empty line for separation
        
        return "\n".join(formatted)

# ================== Enhanced ReWOO Orchestrator ==================

class EnhancedReWOOOrchestrator:
    """
    Enhanced ReWOO orchestrator with better integration and error handling.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, 
                 conversation_history: List[BaseMessage] = None):
        # Store system prompt
        self.system_prompt = SYSTEM_PROMPT
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                system_instruction=self.system_prompt  # Pass system prompt to LLM
            )
        else:
            self.llm = llm
        
        # Initialize components with conversation context
        self.tool_registry = EnhancedToolRegistry(conversation_history)
        self.planner = ContextAwarePlannerAgent(self.llm, self.tool_registry)
        self.worker = RobustWorkerAgent(self.tool_registry)
        self.solver = EnhancedSolverAgent(self.llm)
        
        # Simple in-memory cache
        self.cache = {}
        
        logger.info("Enhanced ReWOO Orchestrator initialized with system prompt constraints")
    
    def update_conversation_history(self, history: List[BaseMessage]):
        """Update the conversation history for context-aware planning."""
        self.tool_registry.conversation_history = history
    
    def process_query(self, query: str, use_cache: bool = True) -> ExecutionResult:
        """Process query with enhanced error handling and context awareness."""
        start_time = datetime.now()
        
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if use_cache and query_hash in self.cache:
            logger.info("Returning cached result")
            return self.cache[query_hash]
        
        try:
            # Step 1: Enhanced Planning
            logger.info("=== ENHANCED PLANNING PHASE ===")
            plan = self.planner.create_plan(query)
            
            # Step 2: Robust Execution
            logger.info("=== ROBUST EXECUTION PHASE ===")
            tool_outputs = self._execute_plan_sequentially(plan)
            
            # Step 3: Enhanced Synthesis
            logger.info("=== ENHANCED SYNTHESIS PHASE ===")
            final_answer = self.solver.synthesize(
                query, tool_outputs, plan.final_synthesis_prompt, plan
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                query=query,
                plan=plan,
                final_answer=final_answer,
                execution_time=execution_time,
                tool_outputs=tool_outputs,
                success=True
            )
            
            # Cache successful results
            if use_cache:
                self.cache[query_hash] = result
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                query=query,
                plan=ExecutionPlan(query=query, steps=[], final_synthesis_prompt=""),
                final_answer=f"I encountered an error processing your request: {str(e)}",
                execution_time=execution_time,
                tool_outputs={},
                success=False,
                error=str(e)
            )
    
    def _execute_plan_sequentially(self, plan: ExecutionPlan) -> Dict[int, Any]:
        """Execute plan sequentially to avoid threading issues."""
        context = {}
        
        for step in plan.steps:
            step_id, result = self.worker.execute_step(step, context)
            context[step_id] = result
            
            # Log step completion
            if step.status == "completed":
                logger.info(f"Step {step_id} completed successfully")
            else:
                logger.warning(f"Step {step_id} failed: {step.error}")
        
        return context