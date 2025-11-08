import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from dotenv import load_dotenv

# Import your existing tools
from tools import ( 
    FindDrug, FindProteinsFromDrug, TextToAQL, 
    PlotSmiles2D, PlotSmiles3D, PredictBindingAffinity,
    GetAminoAcidSequence, GetChemBERTaEmbeddings,
    PreparePDBData, GenerateCompounds, FindSimilarDrugs,
    AnalyzeProtein, PredictADMETProperties,
    PredictDisorderRegionsinProteins, AnalyzeProteinConservation,
    PredictDrugDrugInteractions, PredictLigandBindingSites, 
    EnumerateTautomersAndStereoisomers, GenerateContactMapFromPDB,
    PredictCYP450Sites, PredictBloodBrainBarrierPenetration,
    PredictVeberRules, AutoExtractQSARFeatures, PredictSyntheticAccessibility
)

load_dotenv()

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
    status: str = "pending"
    error: Optional[str] = None
    execution_time: Optional[float] = None
    workflow_stage: Optional[str] = None  # Flexible string instead of enum

@dataclass
class ExecutionPlan:
    """Represents the complete execution plan for a query."""
    query: str
    steps: List[ToolCall]
    final_synthesis_prompt: str
    is_drug_discovery_workflow: bool = False  # Simple boolean flag
    workflow_description: str = ""  # Human-readable workflow description
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
    workflow_summary: Optional[Dict] = None
    error: Optional[str] = None

# ================== Enhanced Tool Registry ==================

class EnhancedToolRegistry:
    """Enhanced tool registry with all available tools."""
    
    def __init__(self, conversation_history: List[BaseMessage] = None):
        self.tools = self._initialize_tools()
        self.tool_descriptions = self._generate_descriptions()
        self.conversation_history = conversation_history or []
        
    def _initialize_tools(self) -> Dict[str, callable]:
        """Initialize all available tools."""
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
        """Generate comprehensive tool descriptions for the planner."""
        tool_categories = {
            "🔍 Drug & Target Discovery": {
                "FindDrug": "Search for detailed drug information by name, returning SMILES, molecular data, and properties",
                "FindProteinsFromDrug": "Find proteins that interact with a specific drug compound",
                "TextToAQL": "Execute complex biomedical database queries using natural language - versatile for broad research questions"
            },
            "🧬 Protein Analysis": {
                "AnalyzeProtein": "Comprehensive protein analysis including structure, function, and domains",
                "GetAminoAcidSequence": "Retrieve amino acid sequences for specific proteins",
                "PredictDisorderRegionsinProteins": "Identify intrinsically disordered regions in protein structures",
                "AnalyzeProteinConservation": "Analyze evolutionary conservation patterns across species",
                "PredictLigandBindingSites": "Identify potential ligand binding pockets on proteins",
                "PreparePDBData": "Prepare and process protein structure data from PDB files",
                "GenerateContactMapFromPDB": "Create residue-residue contact maps from protein structures"
            },
            "⚗️ Compound Generation & Analysis": {
                "GenerateCompounds": "Generate novel chemical compounds with desired properties using AI models",
                "FindSimilarDrugs": "Find structurally similar drugs to a given compound",
                "EnumerateTautomersAndStereoisomers": "Generate all tautomers and stereoisomers of a compound",
                "GetChemBERTaEmbeddings": "Generate molecular embeddings for chemical compounds",
                "PlotSmiles2D": "Generate 2D molecular structure visualizations from SMILES",
                "PlotSmiles3D": "Generate 3D molecular structure visualizations from SMILES"
            },
            "🎯 Binding & Interaction Prediction": {
                "PredictBindingAffinity": "Predict binding affinity between drugs and protein targets",
            },
            "💊 ADMET & Drug Properties": {
                "PredictADMETProperties": "Predict Absorption, Distribution, Metabolism, Excretion, and Toxicity",
                "PredictCYP450Sites": "Predict CYP450 enzyme metabolism sites on compounds",
                "PredictBloodBrainBarrierPenetration": "Predict blood-brain barrier penetration capability",
                "PredictVeberRules": "Evaluate drug-likeness based on Veber's oral bioavailability rules",
                "PredictSyntheticAccessibility": "Predict synthetic accessibility and feasibility",
                "AutoExtractQSARFeatures": "Automatically extract QSAR molecular descriptors"
            },
            "⚠️ Safety & Interactions": {
                "PredictDrugDrugInteractions": "Predict potential interactions between multiple drugs"
            }
        }
        
        descriptions = []
        for category, tools in tool_categories.items():
            descriptions.append(f"\n{category}:")
            for tool_name, desc in tools.items():
                descriptions.append(f"  • {tool_name}: {desc}")
        
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, tool_input: Any) -> Any:
        """Execute a tool with proper error handling."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in registry")
        
        tool_func = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        try:
            if isinstance(tool_input, dict):
                result = tool_func(**tool_input)
            else:
                result = tool_func(tool_input)
                
            logger.info(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "error": True,
                "message": error_msg,
                "tool": tool_name,
                "input": tool_input
            }

# ================== Smart Planner Agent ==================

SYSTEM_PROMPT = """You are an expert AI assistant for drug discovery and pharmaceutical research. You have access to specialized computational tools for analyzing drugs, proteins, and their interactions.

CRITICAL RULES:
1. You can ONLY use the tools explicitly provided to you
2. NO web search, NO internet access, NO external databases
3. Base responses ONLY on tool results you receive
4. Be honest about limitations when tools cannot provide requested information

TOOL SELECTION PHILOSOPHY:
• Choose the MINIMUM number of tools needed to answer the query
• Prefer specific tools (FindDrug, PredictBindingAffinity) for focused questions
• Use TextToAQL for broad exploratory questions or when specific tools don't fit
• For complex drug discovery workflows, intelligently chain multiple tools
• ALWAYS consider conversation history to avoid redundant tool calls

DRUG DISCOVERY WORKFLOW AWARENESS:
You understand the complete drug discovery pipeline:
1. TARGET IDENTIFICATION: Validate protein targets (AnalyzeProtein, PredictDisorderRegionsinProteins, AnalyzeProteinConservation)
2. HIT DISCOVERY: Find/generate candidates (FindDrug, GenerateCompounds, FindSimilarDrugs, EnumerateTautomersAndStereoisomers)
3. BINDING ASSESSMENT: Evaluate interactions (PredictBindingAffinity, PredictLigandBindingSites, PreparePDBData, GenerateContactMapFromPDB)
4. OPTIMIZATION: Refine properties (PredictADMETProperties, PlotSmiles2D/3D, GetChemBERTaEmbeddings, AutoExtractQSARFeatures, PredictVeberRules, PredictSyntheticAccessibility)
5. SAFETY CHECKS: Assess risks (PredictDrugDrugInteractions, PredictCYP450Sites, PredictBloodBrainBarrierPenetration)

BUT: Only trigger multi-stage workflows when users explicitly request comprehensive analysis. For simple queries, use minimal tools."""

class SmartPlannerAgent:
    """Intelligent planner that adapts to query complexity."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI, tool_registry: EnhancedToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_planning_prompt()
    
    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create adaptive planning prompt."""
        return ChatPromptTemplate.from_template("""
You are an expert drug discovery planner. Analyze the query and create an EFFICIENT execution plan.

CONVERSATION HISTORY:
{conversation_history}

AVAILABLE TOOLS (CATEGORIZED):
{tool_descriptions}

USER QUERY: {query}

PLANNING STRATEGY:
1. Determine query complexity: Simple (1-2 tools) vs Complex (multi-stage workflow)
2. For SIMPLE queries: Use minimum tools needed
   - "Tell me about aspirin" → FindDrug only
   - "Show aspirin structure" → FindDrug + PlotSmiles2D
   - "What proteins interact with aspirin" → FindProteinsFromDrug or TextToAQL
   
3. For COMPLEX workflows (user explicitly requests comprehensive analysis):
   - "Complete drug discovery analysis for aspirin against COX-2"
   - "Full ADMET evaluation of compound X"
   - Chain multiple stages with proper dependencies

4. Consider conversation context - don't repeat recent searches

5. Keep tool_input SIMPLE:
   - Strings for most tools: "aspirin", "COX-2", "CC(=O)Oc1ccccc1C(=O)O"
   - Dicts only when required: {{"drug": "aspirin", "protein": "COX-2"}}
   - Use #step_id for dependencies ONLY when necessary

WORKFLOW DETECTION KEYWORDS:
• "comprehensive", "full analysis", "complete workflow" → Multi-stage pipeline
• "end-to-end", "drug discovery workflow" → Multi-stage pipeline  
• "just", "only", "simple", "quick" → Single tool
• Default: Minimal tools unless complexity is clear

OUTPUT FORMAT (JSON):
{{
    "workflow_type": "simple|multi_stage_workflow",
    "workflow_description": "Brief description of what will be done",
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "ToolName",
            "tool_input": "simple_value",
            "dependencies": [],
            "workflow_stage": "stage_name_if_applicable",
            "reasoning": "Why this tool"
        }}
    ],
    "synthesis_prompt": "How to present results"
}}

EXAMPLES:

SIMPLE QUERY: "What is aspirin?"
{{
    "workflow_type": "simple",
    "workflow_description": "Retrieve basic drug information",
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "FindDrug",
            "tool_input": "aspirin",
            "dependencies": [],
            "reasoning": "Get comprehensive aspirin data"
        }}
    ],
    "synthesis_prompt": "Present aspirin information clearly with key properties"
}}

COMPLEX QUERY: "Comprehensive drug discovery analysis of aspirin for COX-2 inhibition"
{{
    "workflow_type": "multi_stage_workflow",
    "workflow_description": "Full drug discovery pipeline: target validation, binding assessment, ADMET evaluation",
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "AnalyzeProtein",
            "tool_input": "COX-2",
            "dependencies": [],
            "workflow_stage": "target_identification",
            "reasoning": "Validate COX-2 as drug target"
        }},
        {{
            "step_id": 2,
            "tool_name": "FindDrug",
            "tool_input": "aspirin",
            "dependencies": [],
            "workflow_stage": "hit_discovery",
            "reasoning": "Get aspirin molecular data"
        }},
        {{
            "step_id": 3,
            "tool_name": "PredictBindingAffinity",
            "tool_input": {{"drug": "aspirin", "protein": "COX-2"}},
            "dependencies": [1, 2],
            "workflow_stage": "binding_assessment",
            "reasoning": "Assess aspirin-COX-2 binding"
        }},
        {{
            "step_id": 4,
            "tool_name": "PredictADMETProperties",
            "tool_input": "#2",
            "dependencies": [2],
            "workflow_stage": "optimization",
            "reasoning": "Evaluate drug-like properties"
        }}
    ],
    "synthesis_prompt": "Compile comprehensive drug discovery report with target validation, binding analysis, and ADMET profile"
}}

Generate the plan now:
""")
    
    def create_plan(self, query: str) -> ExecutionPlan:
        """Create adaptive execution plan based on query complexity."""
        logger.info(f"Creating adaptive plan for: {query}")
        
        # Format conversation history
        history_text = self._format_conversation_history()
        
        try:
            response = self.llm.invoke(
                self.prompt.format(
                    conversation_history=history_text,
                    tool_descriptions=self.tool_registry.tool_descriptions,
                    query=query
                )
            )
            
            # Parse response
            plan_data = self._parse_llm_response(response.content)
            
            # Convert to ExecutionPlan
            steps = []
            for step_data in plan_data["steps"]:
                tool_call = ToolCall(
                    step_id=step_data["step_id"],
                    tool_name=step_data["tool_name"],
                    tool_input=step_data["tool_input"],
                    dependencies=step_data.get("dependencies", []),
                    workflow_stage=step_data.get("workflow_stage")
                )
                steps.append(tool_call)
            
            is_workflow = plan_data.get("workflow_type") == "multi_stage_workflow"
            
            plan = ExecutionPlan(
                query=query,
                steps=steps,
                final_synthesis_prompt=plan_data.get("synthesis_prompt", "Provide clear answer"),
                is_drug_discovery_workflow=is_workflow,
                workflow_description=plan_data.get("workflow_description", "")
            )
            
            logger.info(f"Created {'multi-stage workflow' if is_workflow else 'simple'} plan with {len(steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}, using fallback")
            return self._create_fallback_plan(query)
    
    def _format_conversation_history(self) -> str:
        """Format recent conversation history."""
        if not self.tool_registry.conversation_history:
            return "No previous conversation"
        
        history_text = ""
        recent = self.tool_registry.conversation_history[-6:]  # Last 3 exchanges
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            history_text += f"{role}: {content}\n"
        return history_text
    
    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response with robust error handling."""
        # Clean JSON from markdown
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    def _create_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create intelligent fallback plan based on keywords."""
        query_lower = query.lower()
        
        # Keyword-based tool selection
        if any(kw in query_lower for kw in ["drug", "compound", "molecule", "medicine"]):
            # Extract potential drug name
            words = [w.strip('.,!?') for w in query.split()]
            drug_name = next((w for w in words if len(w) > 3 and w.isalpha()), query)
            
            return ExecutionPlan(
                query=query,
                steps=[ToolCall(step_id=1, tool_name="FindDrug", tool_input=drug_name)],
                final_synthesis_prompt="Present drug information clearly"
            )
        
        elif any(kw in query_lower for kw in ["protein", "enzyme", "target"]):
            return ExecutionPlan(
                query=query,
                steps=[ToolCall(step_id=1, tool_name="TextToAQL", tool_input=query)],
                final_synthesis_prompt="Present protein information clearly"
            )
        
        else:
            # Default to TextToAQL for general queries
            return ExecutionPlan(
                query=query,
                steps=[ToolCall(step_id=1, tool_name="TextToAQL", tool_input=query)],
                final_synthesis_prompt="Provide comprehensive answer"
            )

# ================== Robust Worker Agent ==================

class RobustWorkerAgent:
    """Executes tool calls with proper dependency resolution."""
    
    def __init__(self, tool_registry: EnhancedToolRegistry):
        self.tool_registry = tool_registry
    
    def execute_step(self, step: ToolCall, context: Dict[int, Any]) -> Tuple[int, Any]:
        """Execute a single step with error handling."""
        logger.info(f"Executing Step {step.step_id}: {step.tool_name}")
        
        try:
            # Resolve dependencies
            resolved_input = self._resolve_input(step.tool_input, context)
            
            # Execute tool
            start_time = datetime.now()
            result = self.tool_registry.execute_tool(step.tool_name, resolved_input)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check for errors
            if isinstance(result, dict) and result.get("error"):
                step.status = "failed"
                step.error = result["message"]
                logger.error(f"Step {step.step_id} failed: {result['message']}")
                return (step.step_id, None)
            
            step.output = result
            step.status = "completed"
            step.execution_time = execution_time
            
            logger.info(f"Step {step.step_id} completed in {execution_time:.2f}s")
            return (step.step_id, result)
            
        except Exception as e:
            error_msg = f"Step {step.step_id} exception: {str(e)}"
            logger.error(error_msg)
            step.status = "failed"
            step.error = str(e)
            return (step.step_id, None)
    
    def _resolve_input(self, input_spec: Any, context: Dict[int, Any]) -> Any:
        """Resolve input dependencies like #1 or #2.smiles."""
        if not isinstance(input_spec, str):
            return input_spec
        
        if not input_spec.startswith("#"):
            return input_spec
        
        try:
            # Handle #1.attribute or just #1
            if "." in input_spec:
                parts = input_spec[1:].split(".", 1)
                step_id = int(parts[0])
                attribute = parts[1]
                
                if step_id not in context or context[step_id] is None:
                    logger.warning(f"Step {step_id} not found, returning original")
                    return input_spec
                
                result = context[step_id]
                
                # Navigate nested attributes
                for attr in attribute.split("."):
                    if isinstance(result, dict):
                        result = result.get(attr)
                    else:
                        result = getattr(result, attr, None)
                    
                    if result is None:
                        return input_spec
                
                return result
            else:
                # Simple reference like #1
                step_id = int(input_spec[1:])
                return context.get(step_id, input_spec)
                
        except Exception as e:
            logger.warning(f"Failed to resolve {input_spec}: {e}")
            return input_spec

# ================== Enhanced Solver Agent ==================

class EnhancedSolverAgent:
    """Synthesizes results into comprehensive answers."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_solver_prompt()
    
    def _create_solver_prompt(self) -> ChatPromptTemplate:
        """Create synthesis prompt."""
        return ChatPromptTemplate.from_template(f"""
{SYSTEM_PROMPT}

SYNTHESIS TASK:
Synthesize tool results into a comprehensive answer.

QUERY: {{query}}

WORKFLOW TYPE: {{workflow_type}}
WORKFLOW DESCRIPTION: {{workflow_description}}

TOOL RESULTS:
{{tool_outputs}}

SYNTHESIS INSTRUCTIONS: {{synthesis_prompt}}

SYNTHESIS GUIDELINES:
1. Use ONLY information from tool results (local sources)
2. Structure response appropriately:
   - Simple queries: Direct, concise answers
   - Multi-stage workflows: Comprehensive reports with sections
3. Highlight key scientific findings
4. Acknowledge tool failures or limitations honestly
5. Format molecular data, sequences, and structures clearly
6. For drug discovery workflows, provide actionable insights

FORMATTING:
- Use clear sections for multi-stage workflows
- Include key metrics and predictions
- Reference visualizations when available
- Provide next-step recommendations for workflows

Generate comprehensive response:
""")
    
    def synthesize(self, query: str, tool_outputs: Dict[int, Any], 
                   synthesis_prompt: str, plan: ExecutionPlan) -> str:
        """Synthesize final answer."""
        logger.info("Synthesizing final answer")
        
        formatted_outputs = self._format_outputs(tool_outputs, plan)
        
        workflow_type = "Multi-Stage Drug Discovery Workflow" if plan.is_drug_discovery_workflow else "Simple Query"
        
        try:
            response = self.llm.invoke(
                self.prompt.format(
                    query=query,
                    workflow_type=workflow_type,
                    workflow_description=plan.workflow_description or "N/A",
                    tool_outputs=formatted_outputs,
                    synthesis_prompt=synthesis_prompt
                )
            )
            return response.content
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"""I encountered an error synthesizing results: {str(e)}

Tool outputs:
{formatted_outputs}

Note: I can only use local tools and do not have web access."""
    
    def _format_outputs(self, outputs: Dict[int, Any], plan: ExecutionPlan) -> str:
        """Format tool outputs for synthesis."""
        formatted = []
        
        if plan.is_drug_discovery_workflow:
            # Group by workflow stage
            stages = defaultdict(list)
            for step in plan.steps:
                stage = step.workflow_stage or "general"
                stages[stage].append(step)
            
            for stage_name, steps in stages.items():
                formatted.append(f"\n{'='*60}")
                formatted.append(f"STAGE: {stage_name.upper().replace('_', ' ')}")
                formatted.append(f"{'='*60}\n")
                
                for step in steps:
                    formatted.append(self._format_step(step, outputs.get(step.step_id)))
        else:
            # Simple linear format
            for step in plan.steps:
                formatted.append(self._format_step(step, outputs.get(step.step_id)))
        
        return "\n".join(formatted)
    
    def _format_step(self, step: ToolCall, output: Any) -> str:
        """Format single step output."""
        lines = [
            f"Step {step.step_id}: {step.tool_name}",
            f"  Input: {step.tool_input}",
            f"  Status: {step.status}"
        ]
        
        if step.status == "completed" and output is not None:
            output_str = str(output)
            if len(output_str) > 800:
                output_str = output_str[:800] + "... [truncated]"
            lines.append(f"  Result: {output_str}")
            
            if step.execution_time:
                lines.append(f"  Time: {step.execution_time:.2f}s")
        
        elif step.status == "failed":
            lines.append(f"  Error: {step.error or 'Unknown error'}")
        
        lines.append("")  # Empty line
        return "\n".join(lines)

# ================== Main Orchestrator ==================

class DrugDiscoveryOrchestrator:
    """
    Main orchestrator for drug discovery workflows.
    Handles both simple queries and complex multi-stage pipelines.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None,
                 conversation_history: List[BaseMessage] = None):
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                system_instruction=SYSTEM_PROMPT
            )
        else:
            self.llm = llm
        
        # Initialize components
        self.tool_registry = EnhancedToolRegistry(conversation_history)
        self.planner = SmartPlannerAgent(self.llm, self.tool_registry)
        self.worker = RobustWorkerAgent(self.tool_registry)
        self.solver = EnhancedSolverAgent(self.llm)
        
        # Cache
        self.cache = {}
        
        logger.info("Drug Discovery Orchestrator initialized")
    
    def update_conversation_history(self, history: List[BaseMessage]):
        """Update conversation history."""
        self.tool_registry.conversation_history = history
    
    def process_query(self, query: str, use_cache: bool = True) -> ExecutionResult:
        """
        Process any query - from simple to complex workflows.
        """
        start_time = datetime.now()
        
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if use_cache and query_hash in self.cache:
            logger.info("Returning cached result")
            return self.cache[query_hash]
        
        try:
            # Phase 1: Adaptive Planning
            logger.info("=== ADAPTIVE PLANNING PHASE ===")
            plan = self.planner.create_plan(query)
            logger.info(f"Plan type: {'Multi-stage workflow' if plan.is_drug_discovery_workflow else 'Simple query'}")
            logger.info(f"Steps: {len(plan.steps)}")
            
            # Phase 2: Sequential Execution
            logger.info("=== EXECUTION PHASE ===")
            tool_outputs = self._execute_plan(plan)
            
            # Phase 3: Generate summary for workflows
            workflow_summary = None
            if plan.is_drug_discovery_workflow:
                logger.info("=== WORKFLOW SUMMARY GENERATION ===")
                workflow_summary = self._generate_workflow_summary(plan, tool_outputs)
            
            # Phase 4: Synthesis
            logger.info("=== SYNTHESIS PHASE ===")
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
                workflow_summary=workflow_summary,
                success=True
            )
            
            # Cache result
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
                final_answer=f"I encountered an error: {str(e)}\n\nPlease try rephrasing your query.",
                execution_time=execution_time,
                tool_outputs={},
                success=False,
                error=str(e)
            )
    
    def _execute_plan(self, plan: ExecutionPlan) -> Dict[int, Any]:
        """Execute plan steps sequentially."""
        context = {}
        
        for step in plan.steps:
            step_id, result = self.worker.execute_step(step, context)
            context[step_id] = result
            
            # Log progress
            stage_info = f" [{step.workflow_stage}]" if step.workflow_stage else ""
            if step.status == "completed":
                logger.info(f"✓ Step {step_id} completed{stage_info}")
            else:
                logger.warning(f"✗ Step {step_id} failed{stage_info}: {step.error}")
        
        return context
    
    def _generate_workflow_summary(self, plan: ExecutionPlan, 
                                   tool_outputs: Dict[int, Any]) -> Dict:
        """Generate structured summary for drug discovery workflows."""
        summary = {
            "workflow_description": plan.workflow_description,
            "total_steps": len(plan.steps),
            "successful_steps": sum(1 for s in plan.steps if s.status == "completed"),
            "failed_steps": sum(1 for s in plan.steps if s.status == "failed"),
            "total_execution_time": sum(s.execution_time or 0 for s in plan.steps),
            "stages": {}
        }
        
        # Group by workflow stage
        stage_groups = defaultdict(list)
        for step in plan.steps:
            if step.workflow_stage:
                stage_groups[step.workflow_stage].append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "status": step.status,
                    "execution_time": step.execution_time,
                    "has_output": tool_outputs.get(step.step_id) is not None
                })
        
        summary["stages"] = dict(stage_groups)
        
        return summary
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools with descriptions."""
        tool_info = {}
        for tool_name in self.tool_registry.tools.keys():
            tool_info[tool_name] = f"Available for use"
        return tool_info
    
    def get_statistics(self) -> Dict:
        """Get orchestrator statistics."""
        if not self.cache:
            return {
                "total_queries": 0,
                "simple_queries": 0,
                "workflow_queries": 0,
                "average_execution_time": 0,
                "success_rate": 0
            }
        
        total = len(self.cache)
        workflows = sum(1 for r in self.cache.values() if r.plan.is_drug_discovery_workflow)
        successful = sum(1 for r in self.cache.values() if r.success)
        avg_time = sum(r.execution_time for r in self.cache.values()) / total
        
        return {
            "total_queries": total,
            "simple_queries": total - workflows,
            "workflow_queries": workflows,
            "average_execution_time": round(avg_time, 2),
            "success_rate": round((successful / total) * 100, 2)
        }


# ================== Integration Guide ==================

"""
INTEGRATION GUIDE:
==================

1. BASIC USAGE:
   ```python
   from enhanced_orchestrator import DrugDiscoveryOrchestrator
   
   orchestrator = DrugDiscoveryOrchestrator()
   result = orchestrator.process_query("What is aspirin?")
   print(result.final_answer)
   ```

2. WITH CONVERSATION HISTORY:
   ```python
   from langchain.schema import HumanMessage, AIMessage
   
   history = [
       HumanMessage(content="What is aspirin?"),
       AIMessage(content="Aspirin is..."),
   ]
   
   orchestrator = DrugDiscoveryOrchestrator(conversation_history=history)
   result = orchestrator.process_query("What are its side effects?")
   ```

3. QUERY TYPES SUPPORTED:

   a) Simple Queries:
      - "What is [drug name]?"
      - "Show structure of [compound]"
      - "Find proteins that interact with [drug]"
   
   b) Multi-Tool Queries:
      - "Analyze aspirin and show its 2D structure"
      - "Find aspirin, predict its ADMET properties"
   
   c) Complex Workflows:
      - "Comprehensive drug discovery analysis of [drug] for [target]"
      - "Full ADMET evaluation and optimization of [compound]"
      - "Complete safety assessment of [drug] with interactions"

4. ACCESSING RESULTS:
   ```python
   result = orchestrator.process_query(query)
   
   # Get final answer
   print(result.final_answer)
   
   # Check if it was a workflow
   if result.plan.is_drug_discovery_workflow:
       print(result.workflow_summary)
   
   # Get tool outputs
   for step_id, output in result.tool_outputs.items():
       print(f"Step {step_id}: {output}")
   ```

5. ERROR HANDLING:
   ```python
   result = orchestrator.process_query(query)
   
   if not result.success:
       print(f"Error: {result.error}")
   else:
       print(result.final_answer)
   ```

6. CONVERSATION MANAGEMENT:
   ```python
   # Update history after each interaction
   orchestrator.update_conversation_history(new_history)
   
   # This helps avoid redundant tool calls
   ```

KEY IMPROVEMENTS OVER PREVIOUS VERSIONS:
========================================

1. ADAPTIVE PLANNING:
   - No forced workflows
   - LLM decides complexity
   - Minimal tools for simple queries
   - Full pipelines when explicitly requested

2. FLEXIBLE TOOL SELECTION:
   - Any tool, any combination
   - Works for single-tool queries
   - Handles complex multi-stage workflows
   - Smart dependency resolution

3. ROBUST ERROR HANDLING:
   - Graceful degradation
   - Informative error messages
   - Fallback plans when LLM fails

4. CONVERSATION AWARENESS:
   - Remembers recent exchanges
   - Avoids redundant searches
   - Context-aware responses

5. SIMPLE ARCHITECTURE:
   - No rigid enums or forced stages
   - Clear separation of concerns
   - Easy to debug and extend

WORKFLOW DETECTION:
===================

The orchestrator automatically detects when to use workflows based on:
- Keywords: "comprehensive", "full analysis", "complete", "end-to-end"
- Query complexity: Multiple aspects requested
- Explicit requests: "drug discovery workflow for X"

For simple queries, it uses minimal tools:
- "What is aspirin?" → FindDrug only
- "Show aspirin structure" → FindDrug + PlotSmiles2D
- Default: Smart selection based on query

EXTENDING:
==========

To add new tools:
1. Import the tool function
2. Add to EnhancedToolRegistry._initialize_tools()
3. Add description to _generate_descriptions()
4. Done! LLM will automatically use it

No need to modify pipelines or add enums.
"""

