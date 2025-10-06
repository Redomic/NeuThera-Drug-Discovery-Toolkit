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
from enum import Enum

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
    PredictDrugDrugInteractions, PredictLigandBindingSites, PredictCYP450Sites
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Workflow Stage Enums ==================

class DrugDiscoveryStage(Enum):
    """Represents the different stages in drug discovery pipeline."""
    TARGET_IDENTIFICATION = "target_identification"
    HIT_DISCOVERY = "hit_discovery"
    BINDING_ASSESSMENT = "binding_assessment"
    OPTIMIZATION = "optimization"
    SAFETY_CHECKS = "safety_checks"
    FINAL_SYNTHESIS = "final_synthesis"
    GENERAL_QUERY = "general_query"  # For non-pipeline queries

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
    stage: Optional[DrugDiscoveryStage] = None
    description: str = ""

@dataclass
class ExecutionPlan:
    """Represents the complete execution plan for a query."""
    query: str
    steps: List[ToolCall]
    final_synthesis_prompt: str
    workflow_type: DrugDiscoveryStage = DrugDiscoveryStage.GENERAL_QUERY
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
    workflow_report: Optional[Dict] = None
    error: Optional[str] = None

# ================== Drug Discovery Pipeline Definitions ==================

class DrugDiscoveryPipelines:
    """
    Defines specialized pipelines for each stage of drug discovery.
    """
    
    @staticmethod
    def target_identification_pipeline(target_input: str) -> List[Dict]:
        """
        Stage 1: Target Identification & Validation
        Analyzes protein structure, function, and suitability for binding.
        """
        return [
            {
                "step_id": 1,
                "tool_name": "FindProteinsFromDrug",
                "tool_input": target_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.TARGET_IDENTIFICATION,
                "description": "Find protein targets related to the query",
                "reasoning": "Identify relevant protein targets for analysis"
            },
            {
                "step_id": 2,
                "tool_name": "AnalyzeProtein",
                "tool_input": "#1",
                "dependencies": [1],
                "stage": DrugDiscoveryStage.TARGET_IDENTIFICATION,
                "description": "Comprehensive protein analysis",
                "reasoning": "Analyze protein function and structural properties"
            },
            {
                "step_id": 3,
                "tool_name": "PredictDisorderRegionsinProteins",
                "tool_input": "#1",
                "dependencies": [1],
                "stage": DrugDiscoveryStage.TARGET_IDENTIFICATION,
                "description": "Identify disordered regions",
                "reasoning": "Assess structural suitability for drug binding"
            },
            {
                "step_id": 4,
                "tool_name": "AnalyzeProteinConservation",
                "tool_input": "#1",
                "dependencies": [1],
                "stage": DrugDiscoveryStage.TARGET_IDENTIFICATION,
                "description": "Analyze evolutionary conservation",
                "reasoning": "Validate biological relevance through conservation analysis"
            }
        ]
    
    @staticmethod
    def hit_discovery_pipeline(drug_input: str, generate_novel: bool = False) -> List[Dict]:
        """
        Stage 2: Hit Discovery
        Sources candidate compounds from known drugs or generates novel ones.
        """
        steps = []
        step_id = 1
        
        if generate_novel:
            steps.append({
                "step_id": step_id,
                "tool_name": "GenerateCompounds",
                "tool_input": drug_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.HIT_DISCOVERY,
                "description": "Generate novel drug candidates",
                "reasoning": "Create new molecular structures with desired properties"
            })
            step_id += 1
        else:
            steps.append({
                "step_id": step_id,
                "tool_name": "FindDrug",
                "tool_input": drug_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.HIT_DISCOVERY,
                "description": "Retrieve known drug information",
                "reasoning": "Find existing molecules with relevant properties"
            })
            step_id += 1
        
        steps.append({
            "step_id": step_id,
            "tool_name": "FindSimilarDrugs",
            "tool_input": "#1",
            "dependencies": [1],
            "stage": DrugDiscoveryStage.HIT_DISCOVERY,
            "description": "Find structurally similar compounds",
            "reasoning": "Identify analogs of successful drugs for hit expansion"
        })
        
        return steps
    
    @staticmethod
    def binding_assessment_pipeline(drug_input: str, protein_input: str) -> List[Dict]:
        """
        Stage 3: Binding Assessment
        Evaluates drug-target binding strengths and predicts binding sites.
        """
        return [
            {
                "step_id": 1,
                "tool_name": "PredictLigandBindingSites",
                "tool_input": protein_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.BINDING_ASSESSMENT,
                "description": "Identify potential binding sites",
                "reasoning": "Locate binding pockets for docking predictions"
            },
            {
                "step_id": 2,
                "tool_name": "PreparePDBData",
                "tool_input": protein_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.BINDING_ASSESSMENT,
                "description": "Prepare protein structure data",
                "reasoning": "Ensure proper protein structure for binding analysis"
            },
            {
                "step_id": 3,
                "tool_name": "PredictBindingAffinity",
                "tool_input": {"drug": drug_input, "protein": protein_input},
                "dependencies": [1, 2],
                "stage": DrugDiscoveryStage.BINDING_ASSESSMENT,
                "description": "Calculate binding affinity",
                "reasoning": "Estimate drug-target binding strength"
            }
        ]
    
    @staticmethod
    def optimization_pipeline(compound_input: str) -> List[Dict]:
        """
        Stage 4: Optimization & Property Evaluation
        Refines compounds and evaluates ADMET properties.
        """
        return [
            {
                "step_id": 1,
                "tool_name": "GetChemBERTaEmbeddings",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.OPTIMIZATION,
                "description": "Generate molecular embeddings",
                "reasoning": "Extract features for similarity searches"
            },
            {
                "step_id": 2,
                "tool_name": "PlotSmiles2D",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.OPTIMIZATION,
                "description": "Visualize 2D structure",
                "reasoning": "Provide visual representation of compound"
            },
            {
                "step_id": 3,
                "tool_name": "PlotSmiles3D",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.OPTIMIZATION,
                "description": "Visualize 3D structure",
                "reasoning": "Show three-dimensional conformation"
            },
            {
                "step_id": 4,
                "tool_name": "PredictADMETProperties",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.OPTIMIZATION,
                "description": "Evaluate ADMET properties",
                "reasoning": "Assess Absorption, Distribution, Metabolism, Excretion, and Toxicity"
            },
            {
                "step_id": 5,
                "tool_name": "PredictCYP450Sites",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.OPTIMIZATION,
                "description": "Predict CYP450 metabolism sites",
                "reasoning": "Identify metabolic vulnerability points"
            }
        ]
    
    @staticmethod
    def safety_checks_pipeline(compound_input: str, existing_drugs: List[str] = None) -> List[Dict]:
        """
        Stage 5: Safety & Drug Interaction Checks
        Evaluates potential drug-drug interactions and safety concerns.
        """
        return [
            {
                "step_id": 1,
                "tool_name": "PredictDrugDrugInteractions",
                "tool_input": {"primary_drug": compound_input, "other_drugs": existing_drugs or []},
                "dependencies": [],
                "stage": DrugDiscoveryStage.SAFETY_CHECKS,
                "description": "Predict drug-drug interactions",
                "reasoning": "Identify potential interaction risks with existing therapies"
            },
            {
                "step_id": 2,
                "tool_name": "PredictADMETProperties",
                "tool_input": compound_input,
                "dependencies": [],
                "stage": DrugDiscoveryStage.SAFETY_CHECKS,
                "description": "Validate toxicity profile",
                "reasoning": "Confirm safety and toxicity predictions"
            }
        ]
    
    @staticmethod
    def full_drug_discovery_pipeline(drug_input: str, target_input: str) -> List[Dict]:
        """
        Complete end-to-end drug discovery pipeline.
        Combines all stages from target identification to safety checks.
        """
        pipeline = []
        step_offset = 0
        
        # Stage 1: Target Identification
        target_steps = DrugDiscoveryPipelines.target_identification_pipeline(target_input)
        for step in target_steps:
            step["step_id"] += step_offset
            step["dependencies"] = [d + step_offset for d in step["dependencies"]]
        pipeline.extend(target_steps)
        step_offset += len(target_steps)
        
        # Stage 2: Hit Discovery
        hit_steps = DrugDiscoveryPipelines.hit_discovery_pipeline(drug_input)
        for step in hit_steps:
            step["step_id"] += step_offset
            step["dependencies"] = [d + step_offset for d in step["dependencies"]]
        pipeline.extend(hit_steps)
        step_offset += len(hit_steps)
        
        # Stage 3: Binding Assessment
        binding_steps = DrugDiscoveryPipelines.binding_assessment_pipeline(drug_input, target_input)
        for step in binding_steps:
            step["step_id"] += step_offset
            step["dependencies"] = [d + step_offset for d in step["dependencies"]]
        pipeline.extend(binding_steps)
        step_offset += len(binding_steps)
        
        # Stage 4: Optimization
        opt_steps = DrugDiscoveryPipelines.optimization_pipeline(drug_input)
        for step in opt_steps:
            step["step_id"] += step_offset
            step["dependencies"] = [d + step_offset for d in step["dependencies"]]
        pipeline.extend(opt_steps)
        step_offset += len(opt_steps)
        
        # Stage 5: Safety Checks
        safety_steps = DrugDiscoveryPipelines.safety_checks_pipeline(drug_input)
        for step in safety_steps:
            step["step_id"] += step_offset
            step["dependencies"] = [d + step_offset for d in step["dependencies"]]
        pipeline.extend(safety_steps)
        
        return pipeline

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
            "PredictCYP450Sites": PredictCYP450Sites
        }
    
    def _generate_descriptions(self) -> str:
        """Generate detailed tool descriptions for better planning."""
        descriptions = []
        tool_details = {
            "FindDrug": "Search for detailed drug information by name, returning SMILES, molecular data, and properties",
            "FindProteinsFromDrug": "Find proteins that interact with a specific drug compound",
            "TextToAQL": "Execute complex biomedical database queries using natural language",
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
            "PredictCYP450Sites": "Predict CYP450 metabolism sites on compounds"
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

# ================== Workflow-Aware Planner ==================

SYSTEM_PROMPT = """You are an AI assistant specialized in drug discovery and pharmaceutical research. You orchestrate end-to-end drug discovery workflows.

CRITICAL RULES:
- You can ONLY use the tools that are explicitly provided to you
- No web search, no internet access, no external databases
- Base your responses only on the tool results you receive
- Be honest about your limitations when tools are missing

DRUG DISCOVERY WORKFLOW STAGES:
1. TARGET IDENTIFICATION: Validate protein targets using AnalyzeProtein, PredictDisorderRegionsinProteins, AnalyzeProteinConservation
2. HIT DISCOVERY: Source compounds using FindDrug, GenerateCompounds, FindSimilarDrugs
3. BINDING ASSESSMENT: Evaluate binding using PredictBindingAffinity, PredictLigandBindingSites, PreparePDBData
4. OPTIMIZATION: Refine compounds using ADMET predictions, visualizations, and embeddings
5. SAFETY CHECKS: Assess interactions using PredictDrugDrugInteractions
6. FINAL SYNTHESIS: Compile comprehensive drug discovery report

WORKFLOW DETECTION:
- Recognize when queries require full drug discovery pipelines vs. simple queries
- Identify which stage(s) of the workflow are relevant to the query
- Automatically orchestrate multi-stage workflows for comprehensive drug discovery questions

CONVERSATION CONTEXT:
You have access to our conversation history. Use this context to provide personalized responses."""

class WorkflowAwarePlannerAgent:
    """
    Workflow-aware planner that recognizes drug discovery stages and creates appropriate pipelines.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI, tool_registry: EnhancedToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_planning_prompt()
        self.pipelines = DrugDiscoveryPipelines()
    
    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create a workflow-aware planning prompt."""
        return ChatPromptTemplate.from_template("""
You are an expert drug discovery workflow planner. Analyze the query and create an appropriate execution plan.

CONVERSATION HISTORY:
{conversation_history}

AVAILABLE TOOLS:
{tool_descriptions}

USER QUERY: {query}

WORKFLOW PLANNING STRATEGY:
1. Determine if this requires a drug discovery workflow or a simple query
2. Identify relevant workflow stage(s): Target Identification, Hit Discovery, Binding Assessment, Optimization, Safety Checks
3. For comprehensive queries, orchestrate multi-stage pipelines
4. For specific queries, use targeted single-stage approaches

WORKFLOW KEYWORDS:
- "drug discovery", "find drug for", "develop drug": Full pipeline
- "target", "protein analysis": Target Identification stage
- "generate compounds", "find similar": Hit Discovery stage
- "binding affinity", "docking": Binding Assessment stage
- "ADMET", "toxicity", "optimize": Optimization stage
- "interactions", "safety": Safety Checks stage

Generate a JSON plan with workflow_type:
{{
    "workflow_type": "target_identification|hit_discovery|binding_assessment|optimization|safety_checks|full_pipeline|general_query",
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "ToolName",
            "tool_input": "input_value",
            "dependencies": [],
            "stage": "stage_name",
            "description": "What this accomplishes",
            "reasoning": "Why this tool was chosen"
        }}
    ],
    "synthesis_prompt": "How to compile results into comprehensive drug discovery report"
}}

IMPORTANT:
- Use appropriate workflow pipelines for drug discovery queries
- Keep tool_input values simple (strings or simple dicts)
- Clearly specify the workflow stage for each step
- For full drug discovery queries, include all relevant stages

Generate the plan:
""")
    
    def create_plan(self, query: str) -> ExecutionPlan:
        """Create a workflow-aware execution plan."""
        logger.info(f"Creating workflow-aware execution plan for: {query}")
        
        # Format conversation history
        history_text = ""
        if self.tool_registry.conversation_history:
            recent_messages = self.tool_registry.conversation_history[-4:]
            for msg in recent_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content[:200]}...\n"
        
        # First, try to detect if this is a workflow query
        workflow_type = self._detect_workflow_type(query)
        
        # Generate plan based on workflow type
        if workflow_type != DrugDiscoveryStage.GENERAL_QUERY:
            return self._create_workflow_plan(query, workflow_type)
        
        # For general queries, use LLM-based planning
        response = self.llm.invoke(
            self.prompt.format(
                conversation_history=history_text or "No previous conversation",
                tool_descriptions=self.tool_registry.tool_descriptions,
                query=query
            )
        )
        
        try:
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
                    dependencies=step_data.get("dependencies", []),
                    stage=self._parse_stage(step_data.get("stage")),
                    description=step_data.get("description", "")
                )
                steps.append(tool_call)
            
            workflow_type_str = plan_data.get("workflow_type", "general_query")
            workflow_enum = self._parse_workflow_type(workflow_type_str)
            
            plan = ExecutionPlan(
                query=query,
                steps=steps,
                final_synthesis_prompt=plan_data.get("synthesis_prompt", "Provide a comprehensive answer"),
                workflow_type=workflow_enum
            )
            
            logger.info(f"Created {workflow_enum.value} plan with {len(steps)} steps")
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse planner response: {e}")
            return self._create_smart_fallback_plan(query)
    
    def _detect_workflow_type(self, query: str) -> DrugDiscoveryStage:
        """Detect the appropriate workflow type from the query."""
        query_lower = query.lower()
        
        # Full pipeline keywords
        full_pipeline_keywords = [
            "drug discovery", "find drug for", "develop drug", 
            "discover drug", "end-to-end", "complete analysis"
        ]
        if any(kw in query_lower for kw in full_pipeline_keywords):
            return DrugDiscoveryStage.FINAL_SYNTHESIS  # Will trigger full pipeline
        
        # Stage-specific keywords
        if any(kw in query_lower for kw in ["target", "protein function", "validate target"]):
            return DrugDiscoveryStage.TARGET_IDENTIFICATION
        
        if any(kw in query_lower for kw in ["generate compound", "find similar", "hit discovery"]):
            return DrugDiscoveryStage.HIT_DISCOVERY
        
        if any(kw in query_lower for kw in ["binding", "affinity", "docking"]):
            return DrugDiscoveryStage.BINDING_ASSESSMENT
        
        if any(kw in query_lower for kw in ["admet", "toxicity", "optimize", "metabolism"]):
            return DrugDiscoveryStage.OPTIMIZATION
        
        if any(kw in query_lower for kw in ["interaction", "safety", "drug-drug"]):
            return DrugDiscoveryStage.SAFETY_CHECKS
        
        return DrugDiscoveryStage.GENERAL_QUERY
    
    def _create_workflow_plan(self, query: str, workflow_type: DrugDiscoveryStage) -> ExecutionPlan:
        """Create a plan based on the detected workflow type."""
        # Extract drug and target from query (simplified extraction)
        drug_input = self._extract_entity(query, ["drug", "compound", "molecule"])
        target_input = self._extract_entity(query, ["target", "protein", "enzyme"])
        
        # Get appropriate pipeline steps
        if workflow_type == DrugDiscoveryStage.TARGET_IDENTIFICATION:
            steps_data = self.pipelines.target_identification_pipeline(target_input or query)
        elif workflow_type == DrugDiscoveryStage.HIT_DISCOVERY:
            steps_data = self.pipelines.hit_discovery_pipeline(drug_input or query)
        elif workflow_type == DrugDiscoveryStage.BINDING_ASSESSMENT:
            steps_data = self.pipelines.binding_assessment_pipeline(
                drug_input or query, target_input or query
            )
        elif workflow_type == DrugDiscoveryStage.OPTIMIZATION:
            steps_data = self.pipelines.optimization_pipeline(drug_input or query)
        elif workflow_type == DrugDiscoveryStage.SAFETY_CHECKS:
            steps_data = self.pipelines.safety_checks_pipeline(drug_input or query)
        else:  # FINAL_SYNTHESIS - full pipeline
            steps_data = self.pipelines.full_drug_discovery_pipeline(
                drug_input or query, target_input or query
            )
        
        # Convert to ToolCall objects
        steps = []
        for step_data in steps_data:
            tool_call = ToolCall(
                step_id=step_data["step_id"],
                tool_name=step_data["tool_name"],
                tool_input=step_data["tool_input"],
                dependencies=step_data.get("dependencies", []),
                stage=step_data.get("stage"),
                description=step_data.get("description", "")
            )
            steps.append(tool_call)
        
        synthesis_prompt = self._get_synthesis_prompt_for_workflow(workflow_type)
        
        return ExecutionPlan(
            query=query,
            steps=steps,
            final_synthesis_prompt=synthesis_prompt,
            workflow_type=workflow_type
        )
    
    def _extract_entity(self, query: str, keywords: List[str]) -> str:
        """Simple entity extraction based on keywords."""
        words = query.split()
        for i, word in enumerate(words):
            if any(kw in word.lower() for kw in keywords):
                if i + 1 < len(words):
                    return words[i + 1]
        return query
    
    def _get_synthesis_prompt_for_workflow(self, workflow_type: DrugDiscoveryStage) -> str:
        """Get appropriate synthesis prompt for each workflow type."""
        prompts = {
            DrugDiscoveryStage.TARGET_IDENTIFICATION: """
                Compile a Target Validation Report including:
                - Protein function and structural properties
                - Disordered regions and their implications
                - Conservation analysis and biological relevance
                - Suitability for drug binding
            """,
            DrugDiscoveryStage.HIT_DISCOVERY: """
                Compile a Hit Discovery Report including:
                - Candidate compounds identified
                - Structural analogs and similar drugs
                - Preliminary compound properties
                - Recommendations for further screening
            """,
            DrugDiscoveryStage.BINDING_ASSESSMENT: """
                Compile a Binding Assessment Report including:
                - Predicted binding sites
                - Binding affinity estimates
                - Docking predictions
                - Structural compatibility analysis
            """,
            DrugDiscoveryStage.OPTIMIZATION: """
                Compile an Optimization Report including:
                - ADMET property profile
                - Toxicity predictions
                - Metabolism sites (CYP450)
                - Structural visualizations
                - Medicinal chemistry recommendations
            """,
            DrugDiscoveryStage.SAFETY_CHECKS: """
                Compile a Safety Assessment Report including:
                - Drug-drug interaction predictions
                - Toxicity flags and concerns
                - Safety recommendations
                - Risk mitigation strategies
            """,
            DrugDiscoveryStage.FINAL_SYNTHESIS: """
                Compile a Comprehensive Drug Discovery Report including:
                
                1. TARGET VALIDATION:
                   - Target details and conservation analysis
                   - Structural suitability for binding
                
                2. HIT DISCOVERY:
                   - Candidate compounds identified
                   - Structural analogs explored
                
                3. BINDING ASSESSMENT:
                   - Binding affinity predictions
                   - Docking analysis
                
                4. OPTIMIZATION:
                   - ADMET profiles
                   - Toxicity predictions
                   - Metabolic stability
                
                5. SAFETY:
                   - Drug-drug interaction risks
                   - Safety concerns
                
                6. RECOMMENDATIONS:
                   - Top drug candidates
                   - Next steps in development
                   - Risk factors to address
            """
        }
        return prompts.get(workflow_type, "Provide a comprehensive answer to the user's query")
    
    def _parse_stage(self, stage_str: Optional[str]) -> Optional[DrugDiscoveryStage]:
        """Parse stage string to enum."""
        if not stage_str:
            return None
        try:
            return DrugDiscoveryStage(stage_str)
        except ValueError:
            return None
    
    def _parse_workflow_type(self, workflow_str: str) -> DrugDiscoveryStage:
        """Parse workflow type string to enum."""
        mapping = {
            "target_identification": DrugDiscoveryStage.TARGET_IDENTIFICATION,
            "hit_discovery": DrugDiscoveryStage.HIT_DISCOVERY,
            "binding_assessment": DrugDiscoveryStage.BINDING_ASSESSMENT,
            "optimization": DrugDiscoveryStage.OPTIMIZATION,
            "safety_checks": DrugDiscoveryStage.SAFETY_CHECKS,
            "full_pipeline": DrugDiscoveryStage.FINAL_SYNTHESIS,
            "general_query": DrugDiscoveryStage.GENERAL_QUERY
        }
        return mapping.get(workflow_str, DrugDiscoveryStage.GENERAL_QUERY)
    
    def _create_smart_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create an intelligent fallback plan based on query keywords."""
        query_lower = query.lower()
        
        # Simple keyword-based tool selection
        if any(word in query_lower for word in ["find", "search", "tell me about", "what is"]):
            if any(drug in query_lower for drug in ["drug", "medication", "compound"]):
                tool_name = "FindDrug"
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
                    dependencies=[],
                    stage=DrugDiscoveryStage.GENERAL_QUERY
                )
            ],
            final_synthesis_prompt="Provide a clear and comprehensive answer to the user's question",
            workflow_type=DrugDiscoveryStage.GENERAL_QUERY
        )

# ================== Robust Worker Agent ==================

class RobustWorkerAgent:
    """Enhanced worker agent with better error handling and context preservation."""
    
    def __init__(self, tool_registry: EnhancedToolRegistry):
        self.tool_registry = tool_registry
    
    def execute_step(self, step: ToolCall, context: Dict[int, Any]) -> Tuple[int, Any]:
        """Execute a step with robust error handling."""
        logger.info(f"Executing step {step.step_id}: {step.tool_name} (Stage: {step.stage.value if step.stage else 'N/A'})")
        
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
                    return input_spec
                
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

# ================== Workflow Report Generator ==================

class WorkflowReportGenerator:
    """
    Generates structured reports for drug discovery workflows.
    """
    
    @staticmethod
    def generate_report(plan: ExecutionPlan, tool_outputs: Dict[int, Any]) -> Dict:
        """Generate a structured workflow report."""
        report = {
            "workflow_type": plan.workflow_type.value,
            "query": plan.query,
            "execution_summary": {
                "total_steps": len(plan.steps),
                "successful_steps": sum(1 for s in plan.steps if s.status == "completed"),
                "failed_steps": sum(1 for s in plan.steps if s.status == "failed"),
                "total_execution_time": sum(s.execution_time or 0 for s in plan.steps)
            },
            "stages": {}
        }
        
        # Group results by stage
        stage_results = defaultdict(list)
        for step in plan.steps:
            if step.stage:
                stage_results[step.stage.value].append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "status": step.status,
                    "description": step.description,
                    "output": tool_outputs.get(step.step_id),
                    "execution_time": step.execution_time,
                    "error": step.error
                })
        
        report["stages"] = dict(stage_results)
        
        # Add stage-specific summaries
        if plan.workflow_type == DrugDiscoveryStage.FINAL_SYNTHESIS:
            report["drug_discovery_summary"] = WorkflowReportGenerator._create_full_pipeline_summary(
                stage_results, tool_outputs
            )
        
        return report
    
    @staticmethod
    def _create_full_pipeline_summary(stage_results: Dict, tool_outputs: Dict[int, Any]) -> Dict:
        """Create a summary for full drug discovery pipeline."""
        summary = {
            "target_validation": {},
            "hit_discovery": {},
            "binding_assessment": {},
            "optimization": {},
            "safety_profile": {}
        }
        
        # Extract key information from each stage
        for stage, steps in stage_results.items():
            if stage == DrugDiscoveryStage.TARGET_IDENTIFICATION.value:
                summary["target_validation"] = {
                    "proteins_analyzed": len([s for s in steps if s["status"] == "completed"]),
                    "conservation_data": "Available" if any(s["tool_name"] == "AnalyzeProteinConservation" for s in steps) else "N/A"
                }
            
            elif stage == DrugDiscoveryStage.HIT_DISCOVERY.value:
                summary["hit_discovery"] = {
                    "compounds_found": len([s for s in steps if s["status"] == "completed"]),
                    "similar_drugs": "Available" if any(s["tool_name"] == "FindSimilarDrugs" for s in steps) else "N/A"
                }
            
            elif stage == DrugDiscoveryStage.BINDING_ASSESSMENT.value:
                summary["binding_assessment"] = {
                    "affinity_predicted": any(s["tool_name"] == "PredictBindingAffinity" for s in steps),
                    "binding_sites_identified": any(s["tool_name"] == "PredictLigandBindingSites" for s in steps)
                }
            
            elif stage == DrugDiscoveryStage.OPTIMIZATION.value:
                summary["optimization"] = {
                    "admet_evaluated": any(s["tool_name"] == "PredictADMETProperties" for s in steps),
                    "visualizations_created": any(s["tool_name"] in ["PlotSmiles2D", "PlotSmiles3D"] for s in steps)
                }
            
            elif stage == DrugDiscoveryStage.SAFETY_CHECKS.value:
                summary["safety_profile"] = {
                    "interactions_checked": any(s["tool_name"] == "PredictDrugDrugInteractions" for s in steps),
                    "toxicity_assessed": any(s["tool_name"] == "PredictADMETProperties" for s in steps)
                }
        
        return summary

# ================== Enhanced Solver Agent ==================

class EnhancedSolverAgent:
    """Enhanced solver with workflow-aware synthesis."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = self._create_solver_prompt()
    
    def _create_solver_prompt(self) -> ChatPromptTemplate:
        """Create an enhanced solver prompt."""
        return ChatPromptTemplate.from_template(f"""
{SYSTEM_PROMPT}

SYNTHESIS TASK:
You are synthesizing results from LOCAL TOOLS ONLY for a drug discovery workflow.

WORKFLOW TYPE: {{workflow_type}}

ORIGINAL QUERY: {{query}}

TOOL EXECUTION RESULTS (FROM LOCAL TOOLS ONLY):
{{tool_outputs}}

WORKFLOW REPORT:
{{workflow_report}}

SYNTHESIS INSTRUCTIONS: {{synthesis_prompt}}

Based on the LOCAL tool results above, provide a comprehensive drug discovery response:

CRITICAL SYNTHESIS GUIDELINES:
1. ONLY use information from the local tool results provided
2. Structure your response according to the workflow type
3. Highlight key findings from each stage of the drug discovery process
4. If tools failed, acknowledge limitations and suggest alternatives
5. Be transparent that all data comes from local sources
6. For full pipeline workflows, provide a comprehensive drug discovery report

RESPONSE STRUCTURE FOR DRUG DISCOVERY WORKFLOWS:
- Executive Summary (key findings)
- Stage-by-stage results (Target, Hit Discovery, Binding, Optimization, Safety)
- Key metrics and predictions
- Visualization references
- Risk factors and limitations
- Recommended next steps

IMPORTANT:
- Only use information from the tool results provided
- Format molecular structures, sequences, and data clearly
- Provide actionable insights for drug development
- Be honest about missing information or failed steps

Response:
""")
    
    def synthesize(self, query: str, tool_outputs: Dict[int, Any], 
                   synthesis_prompt: str, execution_plan: ExecutionPlan,
                   workflow_report: Optional[Dict] = None) -> str:
        """Synthesize results with workflow awareness."""
        logger.info(f"Synthesizing final answer for workflow: {execution_plan.workflow_type.value}")
        
        # Format outputs with workflow context
        formatted_outputs = self._format_outputs_enhanced(tool_outputs, execution_plan)
        
        # Format workflow report
        report_str = json.dumps(workflow_report, indent=2) if workflow_report else "No workflow report available"
        
        try:
            response = self.llm.invoke(
                self.prompt.format(
                    workflow_type=execution_plan.workflow_type.value,
                    query=query,
                    tool_outputs=formatted_outputs,
                    workflow_report=report_str,
                    synthesis_prompt=synthesis_prompt
                )
            )
            return response.content
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return f"""I apologize, but I encountered an error while synthesizing the results: {str(e)}

Based on the available local tool outputs:
{formatted_outputs}

Workflow Report:
{report_str}

Please note: I can only use the local tools available to me and do not have web access."""
    
    def _format_outputs_enhanced(self, outputs: Dict[int, Any], plan: ExecutionPlan) -> str:
        """Enhanced output formatting with workflow stage context."""
        formatted = []
        
        # Group by stage
        stage_groups = defaultdict(list)
        for step in plan.steps:
            stage_name = step.stage.value if step.stage else "general"
            stage_groups[stage_name].append(step)
        
        # Format by stage
        for stage_name, steps in stage_groups.items():
            formatted.append(f"\n{'='*60}")
            formatted.append(f"STAGE: {stage_name.upper()}")
            formatted.append(f"{'='*60}")
            
            for step in steps:
                step_id = step.step_id
                output = outputs.get(step_id)
                
                formatted.append(f"\nStep {step_id}: {step.tool_name}")
                formatted.append(f"  Description: {step.description}")
                formatted.append(f"  Input: {step.tool_input}")
                formatted.append(f"  Status: {step.status}")
                
                if step.status == "completed" and output is not None:
                    output_str = str(output)
                    if len(output_str) > 1000:
                        output_str = output_str[:1000] + "... [truncated]"
                    formatted.append(f"  Result: {output_str}")
                elif step.status == "failed":
                    formatted.append(f"  Error: {step.error or 'Unknown error'}")
                else:
                    formatted.append(f"  Result: No output available")
                
                if step.execution_time:
                    formatted.append(f"  Execution Time: {step.execution_time:.2f}s")
        
        return "\n".join(formatted)

# ================== Enhanced ReWOO Orchestrator ==================

class EnhancedReWOOOrchestrator:
    """
    Enhanced ReWOO orchestrator with drug discovery workflow support.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, 
                 conversation_history: List[BaseMessage] = None):
        self.system_prompt = SYSTEM_PROMPT
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                system_instruction=self.system_prompt
            )
        else:
            self.llm = llm
        
        # Initialize components with conversation context
        self.tool_registry = EnhancedToolRegistry(conversation_history)
        self.planner = WorkflowAwarePlannerAgent(self.llm, self.tool_registry)
        self.worker = RobustWorkerAgent(self.tool_registry)
        self.solver = EnhancedSolverAgent(self.llm)
        self.report_generator = WorkflowReportGenerator()
        
        # Simple in-memory cache
        self.cache = {}
        
        logger.info("Enhanced ReWOO Orchestrator initialized with Drug Discovery Workflow support")
    
    def update_conversation_history(self, history: List[BaseMessage]):
        """Update the conversation history for context-aware planning."""
        self.tool_registry.conversation_history = history
    
    def process_query(self, query: str, use_cache: bool = True) -> ExecutionResult:
        """Process query with workflow-aware drug discovery orchestration."""
        start_time = datetime.now()
        
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if use_cache and query_hash in self.cache:
            logger.info("Returning cached result")
            return self.cache[query_hash]
        
        try:
            # Step 1: Workflow-Aware Planning
            logger.info("=== WORKFLOW-AWARE PLANNING PHASE ===")
            plan = self.planner.create_plan(query)
            logger.info(f"Detected workflow type: {plan.workflow_type.value}")
            
            # Step 2: Robust Execution
            logger.info("=== WORKFLOW EXECUTION PHASE ===")
            tool_outputs = self._execute_plan_sequentially(plan)
            
            # Step 3: Generate Workflow Report
            logger.info("=== WORKFLOW REPORT GENERATION ===")
            workflow_report = self.report_generator.generate_report(plan, tool_outputs)
            
            # Step 4: Enhanced Synthesis
            logger.info("=== WORKFLOW-AWARE SYNTHESIS PHASE ===")
            final_answer = self.solver.synthesize(
                query, tool_outputs, plan.final_synthesis_prompt, plan, workflow_report
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                query=query,
                plan=plan,
                final_answer=final_answer,
                execution_time=execution_time,
                tool_outputs=tool_outputs,
                workflow_report=workflow_report,
                success=True
            )
            
            # Cache successful results
            if use_cache:
                self.cache[query_hash] = result
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            logger.info(f"Workflow Summary: {workflow_report['execution_summary']}")
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
            
            # Log step completion with stage info
            stage_info = f" (Stage: {step.stage.value})" if step.stage else ""
            if step.status == "completed":
                logger.info(f"Step {step_id} completed successfully{stage_info}")
            else:
                logger.warning(f"Step {step_id} failed{stage_info}: {step.error}")
        
        return context
    
    def get_workflow_statistics(self) -> Dict:
        """Get statistics about workflows executed."""
        stats = {
            "total_queries": len(self.cache),
            "workflow_types": defaultdict(int),
            "average_execution_time": 0,
            "success_rate": 0
        }
        
        if not self.cache:
            return stats
        
        total_time = 0
        successful = 0
        
        for result in self.cache.values():
            stats["workflow_types"][result.plan.workflow_type.value] += 1
            total_time += result.execution_time
            if result.success:
                successful += 1
        
        stats["average_execution_time"] = total_time / len(self.cache)
        stats["success_rate"] = (successful / len(self.cache)) * 100
        stats["workflow_types"] = dict(stats["workflow_types"])
        
        return stats

