import os
import sys
import ast

import requests
import ast
import json
import hashlib
import tempfile
import re

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from glob import glob
from io import StringIO

from db import db, arango_graph

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from arango import ArangoClient

from transformers import AutoTokenizer, AutoModel
import torch

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.llms.bedrock import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler

from pydantic import BaseModel, Field

from Bio.PDB import MMCIFParser

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

import streamlit as st
import networkx as nx
from pyvis.network import Network

from DeepPurpose import utils
from DeepPurpose import DTI as models

from TamGen_custom import TamGenCustom

#================= Models & DB =================

sys.path.append(os.path.abspath("./TamGen"))

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

drug_collection = db.collection('drug')
link_collection = db.collection('drug-protein') 

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

worker = TamGenCustom(
    data="./TamGen_Demo_Data",
    ckpt="checkpoints/crossdock_pdb_A10/checkpoint_best.pt",
    use_conditional=True
)

# ================== Helper ==================

def _GenerateKey(smiles: str) -> str:
    """Generate a unique _key for the compound using SMILES hash."""
    hash_value = hashlib.sha256(smiles.encode()).hexdigest()[:8]
    return f"GEN:{hash_value}"

def _SanitizeInput(d: Any, list_limit: int) -> Any:
    """Sanitize the input dictionary or list.

    Sanitizes the input by removing embedding-like values,
    lists with more than **list_limit** elements, that are mostly irrelevant for
    generating answers in a LLM context. These properties, if left in
    results, can occupy significant context space and detract from
    the LLM's performance by introducing unnecessary noise and cost.

    Args:
        d (Any): The input dictionary or list to sanitize.
        list_limit (int): The maximum allowed length of lists.

    Returns:
        Any: The sanitized dictionary or list.
    """
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sanitized_value = _SanitizeInput(value, list_limit)
                if sanitized_value is not None:
                    new_dict[key] = sanitized_value
            elif isinstance(value, list):
                if len(value) < list_limit:
                    sanitized_value = _SanitizeInput(value, list_limit)
                    if sanitized_value is not None:
                        new_dict[key] = sanitized_value
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) == 0:
            return d
        elif len(d) < list_limit:
            return [_SanitizeInput(item, list_limit) for item in d if _SanitizeInput(item, list_limit) is not None]
        else:
            return f"List of {len(d)} elements of type {type(d[0])}"
    else:
        return d

def _display_sidebar_output(title: str, content: Any, content_type: str = "json") -> None:
    """Helper function to consistently display outputs in sidebar."""
    with st.sidebar:
        st.markdown(f"**{title}**")
        if content_type == "json":
            st.json(content)
        elif content_type == "code":
            st.code(content, language="aql")
        elif content_type == "success":
            st.success(content)
        elif content_type == "error":
            st.error(content)
        st.divider()

# ================= Enhanced Functions =================

def FindDrug(drug_name: str) -> Optional[Dict]:
    """
    DRUG LOOKUP: Searches the comprehensive drug database for detailed pharmaceutical information.
    
    Use this tool when you need to find:
    - Drug specifications (SMILES, InChI, CAS numbers)
    - Drug identifiers (ChEMBL ID, UNII, accession numbers)
    - Alternative drug names and synonyms
    - Whether a drug was computationally generated
    
    Input: Exact drug name or any known synonym
    Returns: Complete drug profile including molecular representations

    Args:
        drug_name (str): The name of the drug to search for (case-insensitive)

    Returns:
        dict or None: Complete drug information if found, None if not found
    """
    try:
        query = """
        FOR d IN drug
            FILTER LOWER(d.drug_name) == LOWER(@name) OR LOWER(@name) IN TOKENS(d.synonym, "text_en")
            RETURN {
                _id: d._id,
                _key: d._key,
                accession: d.accession,
                drug_name: d.drug_name,
                cas: d.cas,
                unii: d.unii,
                synonym: d.synonym,
                key: d.key,
                chembl: d.chembl,
                smiles: d.smiles,
                inchi: d.inchi,
                generated: d.generated
            }
        """
        
        cursor = db.aql.execute(query, bind_vars={"name": drug_name})
        results = list(cursor)

        if results:
            _display_sidebar_output("Drug Found", results[0])
            return results[0]
        else:
            _display_sidebar_output("Drug Search", f"No results found for: {drug_name}", "error")
            return None
            
    except Exception as e:
        _display_sidebar_output("Database Error", f"Error searching for drug: {str(e)}", "error")
        return None

def FindProteinsFromDrug(drug_name: str) -> List[str]:
    """
    DRUG-PROTEIN NETWORK ANALYSIS: Discovers protein targets associated with a specific drug and visualizes the interaction network.
    
    Use this tool to:
    - Find all proteins that interact with a given drug
    - Understand drug mechanism of action through protein targets
    - Visualize drug-protein-gene interaction networks
    - Identify potential off-target effects
    
    Input: Drug name (must exist in database)
    Output: List of PDB IDs + Interactive network visualization

    Args:
        drug_name (str): The name of the drug to analyze

    Returns:
        List[str]: List of PDB IDs of associated proteins
    """
    try:
        # Query for proteins associated with the drug
        query = """
        FOR d IN drug 
            FILTER LOWER(d.drug_name) == LOWER(@drug_name)
            LIMIT 1  
            FOR v, e, p IN 1..2 OUTBOUND d._id
                GRAPH "NeuThera"
                FILTER IS_SAME_COLLECTION("protein", v)
                LIMIT 10
                RETURN DISTINCT { _key: v._key }
        """

        cursor = db.aql.execute(query, bind_vars={"drug_name": drug_name})
        proteins = [doc["_key"] for doc in cursor]

        if not proteins:
            _display_sidebar_output("Protein Search", f"No proteins found for drug: {drug_name}", "error")
            return []

        # Generate network visualization
        graph_query = """
        FOR d IN drug
            FILTER LOWER(d.drug_name) == LOWER(@drug_name)
            LIMIT 1  
            FOR v, e, p IN 1..3 OUTBOUND d._id GRAPH "NeuThera"
                LIMIT 500
                RETURN { 
                    from: p.vertices[-2]._key,
                    to: v._key,
                    type: PARSE_IDENTIFIER(v._id).collection
                }
        """

        graph_cursor = db.aql.execute(graph_query, bind_vars={"drug_name": drug_name})

        # Create network visualization
        net = Network(height="500px", width="100%", directed=True, notebook=False)
        net.force_atlas_2based()

        nodes = set()
        edges = set()

        # Pattern matching for node classification
        drug_pattern = re.compile(r"^DB\d+$", re.IGNORECASE)
        gene_pattern = re.compile(r"^[A-Z0-9]+$")
        protein_pattern = re.compile(r"^\d\w{3}$")

        def classify_node(node):
            if drug_pattern.match(node):
                return "drug"
            elif protein_pattern.match(node):
                return "protein"
            elif gene_pattern.match(node):
                return "gene"
            return "unknown"

        color_map = {
            "drug": "#5fa8d3",
            "gene": "#a7c957",
            "protein": "#bc4749",
            "unknown": "#999999"
        }

        # Build network
        for doc in graph_cursor:
            if (doc["from"] is not None) and (doc["to"] is not None):
                from_node = doc["from"]
                to_node = doc["to"]
                edge_type = doc["type"]

                nodes.add(from_node)
                nodes.add(to_node)
                edges.add((from_node, to_node, edge_type))

        # Add nodes and edges to network
        for node in nodes:
            net.add_node(node, label=node, color=color_map[classify_node(node)])

        for from_node, to_node, edge_type in edges:
            net.add_edge(from_node, to_node, title=edge_type, color="#5c677d")

        # Save and display network
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as file:
            html = file.read()

        _display_sidebar_output("Network Query", graph_query, "code")
        _display_sidebar_output("Associated Proteins", proteins)

        if proteins:
            st.components.v1.html(html, height=550, scrolling=True)

        return proteins
        
    except Exception as e:
        _display_sidebar_output("Network Analysis Error", f"Error finding proteins: {str(e)}", "error")
        return []

# Prepare sanitized schema for ArangoDB
sanitized_schema = _SanitizeInput(d=arango_graph.schema, list_limit=32)
arango_graph.set_schema(sanitized_schema)

def TextToAQL(query: str) -> str:
    """
    BIOMEDICAL KNOWLEDGE GRAPH QUERY: Converts natural language questions into database queries for comprehensive biomedical data retrieval.
    
    This tool accesses a specialized knowledge graph containing:
    - Drug-protein interactions and binding data
    - Gene-disease associations and pathways
    - Protein structural information
    - Drug mechanism pathways
    - Clinical trial and pharmacological data
    
    Use this tool when you need to:
    - Explore complex relationships in biomedical data
    - Find connections that specific tools don't cover
    - Query broad patterns or associations
    - Access comprehensive database information
    
    Examples of suitable queries:
    - "What diseases are associated with gene X?"
    - "Which drugs target protein Y?"
    - "Find drugs related to PDB Z"
    - "Show me the interaction network for compound A"
    - "What are the pathways involved in condition B?"
    
    Input: Natural language question about biomedical relationships
    Output: Structured answer from the knowledge graph

    Args:
        query (str): Natural language question about biomedical data

    Returns:
        str: Formatted answer based on knowledge graph data
    """
    try:
        # Use Gemini 2.0 Flash for GraphRAG
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0,
            google_api_key=google_api_key,
            max_retries=3
        )

        class EnhancedCallbackHandler(BaseCallbackHandler):
            def on_text(self, text: str, **kwargs) -> None:
                if isinstance(text, str):
                    if "AQL Query" in text:
                        pass  # Will be handled by sidebar display
                    elif "AQL Result" in text:
                        pass  # Will be handled by sidebar display
                    else:
                        try:
                            parsed = ast.literal_eval(text.strip())
                            if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                                _display_sidebar_output("AQL Result", parsed)
                        except Exception:
                            if "syntax error" in text.lower():
                                _display_sidebar_output("Syntax Error", text.strip(), "error")
                            else:
                                _display_sidebar_output("Generated AQL", text.strip(), "code")

            def on_chain_end(self, *args, **kwargs) -> None:
                result = kwargs.get('outputs', {}).get('result')
                if result:
                    _display_sidebar_output("Final Answer", result, "success")

        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            callbacks=[EnhancedCallbackHandler()]
        )

        result = chain.invoke(query)
        return str(result["result"])
        
    except Exception as e:
        error_msg = f"GraphRAG query failed: {str(e)}"
        _display_sidebar_output("GraphRAG Error", error_msg, "error")
        return error_msg

def PlotSmiles2D(smiles: str) -> bool:
    """
    2D MOLECULAR VISUALIZATION: Generates publication-quality 2D chemical structure diagrams.
    
    Use this tool to:
    - Visualize molecular structures for analysis
    - Compare multiple compounds side by side
    - Generate figures for reports or presentations
    - Verify SMILES string validity
    
    Input: Valid SMILES string
    Output: High-quality 2D molecular structure image
    
    Args:
        smiles (str): SMILES representation of the molecule

    Returns:
        bool: True if successfully plotted, False if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.write(Draw.MolToImage(mol, size=(300, 300))) 
            _display_sidebar_output("2D Structure", f"Successfully rendered structure for: {smiles}", "success")
            return True
        else:
            _display_sidebar_output("Invalid SMILES", f"Cannot parse SMILES: {smiles}", "error")
            return False
    except Exception as e:
        _display_sidebar_output("Rendering Error", f"Failed to render 2D structure: {str(e)}", "error")
        return False

def PlotSmiles3D(smiles: str) -> bool:
    """
    3D MOLECULAR VISUALIZATION: Creates interactive 3D molecular structures with optimized conformations.
    
    Use this tool for:
    - Advanced structural analysis and drug design
    - Understanding molecular geometry and binding sites
    - Interactive exploration of chemical conformations
    - Stereochemistry visualization
    
    Features:
    - Force field optimized geometries (UFF)
    - Interactive rotation and zoom
    - Atom-type color coding
    - Bond visualization
    
    Input: Valid SMILES string
    Output: Interactive 3D molecular model + 2D reference structure

    Args:
        smiles (str): SMILES representation of the molecule

    Returns:
        bool: True if successfully generated 3D structure, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _display_sidebar_output("Invalid SMILES", f"Cannot parse SMILES: {smiles}", "error")
            return False

        # Prepare molecule for 3D
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol) 

        # Generate 3D coordinates
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status == -1: 
            _display_sidebar_output("3D Generation Error", "Failed to generate 3D coordinates", "error")
            return False

        # Optimize geometry
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            _display_sidebar_output("Optimization Warning", "UFF optimization failed, using unoptimized structure", "error")

        conformer = mol.GetConformer()
        if not conformer.Is3D():
            _display_sidebar_output("3D Validation Error", "Generated structure is not 3D", "error")
            return False

        # Extract 3D coordinates and atom information
        atom_positions = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

        # Color coding by atom type
        atom_colors = []
        for atom in atom_symbols:
            if atom == 'O': atom_colors.append('red')
            elif atom == 'N': atom_colors.append('blue')
            elif atom == 'C': atom_colors.append('gray')
            elif atom == 'H': atom_colors.append('lightgray')
            elif atom == 'S': atom_colors.append('yellow')
            else: atom_colors.append('purple')

        # Create 3D plot
        fig = go.Figure()
        
        # Add atoms
        fig.add_trace(go.Scatter3d(
            x=atom_positions[:, 0], y=atom_positions[:, 1], z=atom_positions[:, 2],
            mode='markers+text',
            marker=dict(size=8, color=atom_colors, opacity=0.8),
            text=atom_symbols,
            textposition="top center",
            showlegend=False,
            name="Atoms"
        ))

        # Add bonds
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            fig.add_trace(go.Scatter3d(
                x=[atom_positions[start][0], atom_positions[end][0]],
                y=[atom_positions[start][1], atom_positions[end][1]],
                z=[atom_positions[start][2], atom_positions[end][2]],
                mode='lines',
                line=dict(color='gray', width=3),
                showlegend=False,
                name="Bonds"
            ))

        fig.update_layout(
            title=f"3D Molecular Structure: {smiles}",
            scene=dict(
                xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
                aspectmode='cube'
            ),
            width=700, height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False
        )
        
        # Display 2D reference in sidebar
        _display_sidebar_output("2D Reference Structure", "Generated from same SMILES", "success")
        with st.sidebar:
            st.write(Draw.MolToImage(mol, size=(200, 200)))
            st.divider()
            
        st.write(fig)
        return True
        
    except Exception as e:
        _display_sidebar_output("3D Visualization Error", f"Failed to generate 3D structure: {str(e)}", "error")
        return False

def PredictBindingAffinity(input_data: Union[str, Dict], y: List[float] = [7.635]) -> float:
    """
    AI-POWERED BINDING AFFINITY PREDICTION: Predicts drug-target binding strength using deep learning models.
    
    Use this tool for:
    - Drug discovery and optimization
    - Virtual screening of compound libraries
    - Understanding drug-target interactions
    - Predicting off-target effects
    
    Model: Pre-trained CNN-based DTI (Drug-Target Interaction) predictor
    Output: Binding affinity as log(Kd) or log(Ki) value
    
    Higher values = stronger binding
    Typical range: 4-12 (corresponding to nM to mM dissociation constants)

    Args:
        input_data (Union[str, Dict]): JSON string or dict containing:
            - x_drug (str): SMILES representation of the drug
            - x_target (str): Amino acid sequence of the protein target
        y (List[float]): Reference binding values (default: [7.635])

    Returns:
        float: Predicted binding affinity value
    """
    try:
        if isinstance(input_data, str): 
            input_data = json.loads(input_data)

        x_drug = input_data.get("x_drug")
        x_target = input_data.get("x_target")

        if not x_drug or not x_target:
            error_msg = "Both x_drug (SMILES) and x_target (amino acid sequence) must be provided"
            _display_sidebar_output("Input Error", error_msg, "error")
            raise ValueError(error_msg)

        print(f"Predicting binding affinity for drug: {x_drug[:50]}... target: {x_target[:50]}...")

        X_drug = [x_drug]
        X_target = [x_target]
        
        # Load pre-trained model
        binding_model = models.model_pretrained(path_dir='DTI_model')
        X_pred = utils.data_process(X_drug, X_target, y, drug_encoding='CNN', target_encoding='CNN', split_method='no_split')
        predictions = binding_model.predict(X_pred)

        predicted_affinity = float(predictions[0])
        
        _display_sidebar_output("Binding Affinity Prediction", {
            "predicted_affinity": predicted_affinity,
            "interpretation": "Higher values indicate stronger binding",
            "drug_smiles": x_drug[:50] + "..." if len(x_drug) > 50 else x_drug,
            "target_length": len(x_target)
        })

        return predicted_affinity
        
    except Exception as e:
        error_msg = f"Binding affinity prediction failed: {str(e)}"
        _display_sidebar_output("Prediction Error", error_msg, "error")
        return 0.0

def GetAminoAcidSequence(pdb_id: str) -> Dict[str, str]:    
    """
    PROTEIN SEQUENCE EXTRACTION: Extracts amino acid sequences from PDB structure files for binding affinity predictions.
    
    Use this tool to:
    - Prepare protein sequences for DTI prediction models
    - Extract chain-specific sequences from crystal structures
    - Validate protein structure data availability
    - Support drug-target interaction analysis
    
    Input: Valid PDB ID (e.g., "1ABC", "2xyz")
    Output: Dictionary mapping chain IDs to amino acid sequences
    
    Note: Sequences are automatically formatted for machine learning models

    Args:
        pdb_id (str): 4-character PDB ID of the protein structure

    Returns:
        Dict[str, str]: Dictionary where keys are chain IDs and values are amino acid sequences
    """
    try:
        print(f"Extracting amino acid sequences for PDB: {pdb_id}")

        cif_file_path = f"./database/PDBlib/{pdb_id.lower()}.cif"
        
        if not os.path.exists(cif_file_path):
            error_msg = f"PDB file not found: {cif_file_path}"
            _display_sidebar_output("File Not Found", error_msg, "error")
            return {}

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file_path)
        
        sequences = {}
        for model in structure:
            for chain in model:
                # Extract amino acid sequence (excluding heteroatoms)
                seq = "".join(residue.resname for residue in chain if residue.id[0] == " ")
                if seq:  # Only include chains with actual amino acids
                    sequences[chain.id] = seq 
                    
        if sequences:
            _display_sidebar_output("Sequence Extraction", {
                "pdb_id": pdb_id,
                "chains_found": list(sequences.keys()),
                "total_chains": len(sequences),
                "longest_chain": max(len(seq) for seq in sequences.values()) if sequences else 0
            })
        else:
            _display_sidebar_output("No Sequences", f"No amino acid sequences found in {pdb_id}", "error")
        
        return sequences
        
    except Exception as e:
        error_msg = f"Failed to extract sequences from {pdb_id}: {str(e)}"
        _display_sidebar_output("Extraction Error", error_msg, "error")
        return {}

def GetChemBERTaEmbeddings(smiles: str) -> Optional[List[float]]:
    """
    MOLECULAR EMBEDDING GENERATION: Creates high-dimensional vector representations of molecules using ChemBERTa.
    
    Use this tool for:
    - Molecular similarity searches
    - Machine learning feature generation
    - Chemical space analysis
    - Drug clustering and classification
    
    Model: ChemBERTa (ZINC-trained transformer)
    Output: 768-dimensional vector embedding
    
    These embeddings capture chemical and structural properties for computational analysis.

    Args:
        smiles (str): Valid SMILES representation of a molecule

    Returns:
        List[float] or None: 768-dimensional vector as list of floats, None if invalid input
    """
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            _display_sidebar_output("Invalid Input", "SMILES string cannot be empty", "error")
            return None 

        print(f"Generating ChemBERTa embedding for: {smiles}")

        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).tolist()[0]
        
        _display_sidebar_output("Embedding Generated", {
            "smiles": smiles,
            "embedding_dim": len(embedding),
            "model": "ChemBERTa-ZINC-base-v1"
        })

        return embedding
        
    except Exception as e:
        error_msg = f"Failed to generate embedding: {str(e)}"
        _display_sidebar_output("Embedding Error", error_msg, "error")
        return None

def PreparePDBData(pdb_id: str) -> bool:
    """
    PDB DATA PREPARATION: Downloads and processes protein structure data for molecular generation workflows.
    
    CRITICAL: Always run this before using GenerateCompounds or other PDB-dependent tools.
    
    This tool:
    - Downloads PDB structure files if missing
    - Processes data for TamGen molecular generation
    - Validates structure availability
    - Prepares binding site information
    
    Input: Valid PDB ID (e.g., "1ABC")
    Output: Boolean indicating successful preparation

    Args:
        pdb_id (str): 4-character PDB ID of the target protein structure

    Returns:
        bool: True if data preparation successful, False otherwise
    """
    try:
        DemoDataFolder = "TamGen_Demo_Data"
        ligand_inchi = None
        thr = 10

        out_split = pdb_id.lower()
        FF = glob(f"{DemoDataFolder}/*")
        
        # Check if data already exists
        for ff in FF:
            if f"gen_{out_split}" in ff:
                print(f"PDB data for {pdb_id} already available")
                _display_sidebar_output("PDB Data Status", f"Data for {pdb_id} already prepared", "success")
                return True
        
        print(f"Preparing PDB data for {pdb_id}...")
        os.makedirs(DemoDataFolder, exist_ok=True)
        
        # Create temporary input file
        with open("tmp_pdb.csv", "w") as fw:
            if ligand_inchi is None:
                print("pdb_id", file=fw)
                print(f"{pdb_id}", file=fw)
            else:
                print("pdb_id,ligand_inchi", file=fw)
                print(f"{pdb_id},{ligand_inchi}", file=fw)
        
        _display_sidebar_output("PDB Preparation", f"Processing PDB data for {pdb_id}...", "success")

        # Run preparation script
        script_path = os.path.abspath("TamGen/scripts/build_data/prepare_pdb_ids.py")
        exit_code = os.system(f"python {script_path} tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
        
        # Cleanup
        if os.path.exists("tmp_pdb.csv"):
            os.remove("tmp_pdb.csv")
            
        if exit_code == 0:
            _display_sidebar_output("PDB Preparation Complete", f"Successfully prepared data for {pdb_id}", "success")
            return True
        else:
            _display_sidebar_output("PDB Preparation Failed", f"Failed to prepare data for {pdb_id}", "error")
            return False
            
    except Exception as e:
        error_msg = f"PDB preparation error: {str(e)}"
        _display_sidebar_output("Preparation Error", error_msg, "error")
        return False

def GenerateCompounds(pdb_id: str) -> Dict[str, Union[str, List[str]]]:
    """
    AI-POWERED MOLECULAR GENERATION: Creates novel drug-like compounds optimized for specific protein targets.
    
    Use this tool for:
    - De novo drug design and lead optimization
    - Generating focused compound libraries
    - Structure-based drug discovery
    - Lead compound diversification
    
    Features:
    - Target-specific molecular generation using TamGen
    - Similarity-based ranking against reference compounds
    - Automatic database integration for further analysis
    - Visual compound grid display
    
    Prerequisites: Must run PreparePDBData(pdb_id) first
    
    Input: Valid PDB ID of target protein
    Output: Reference compound + generated analogs ranked by similarity

    Args:
        pdb_id (str): 4-character PDB ID of the target protein

    Returns:
        Dict[str, Union[str, List[str]]]: Dictionary containing reference_smile and generated_smiles
    """
    try:
        num_samples = 3
        max_seed = 5

        print(f"Generating compounds for PDB: {pdb_id}")
        
        # Load target-specific data
        worker.reload_data(subset=f"gen_{pdb_id.lower()}")

        print(f"Generating {num_samples} target-optimized compounds...")
        _display_sidebar_output("Generation Status", f"Generating {num_samples} compounds for {pdb_id}...", "success")

        # Generate molecules
        generated_mols, reference_mol = worker.sample(
            m_sample=num_samples, 
            maxseed=max_seed
        )

        if not generated_mols:
            error_msg = "No compounds generated"
            _display_sidebar_output("Generation Failed", error_msg, "error")
            return {"error": error_msg}

        # Process reference molecule
        reference_smile = ""
        if reference_mol:
            if isinstance(reference_mol, str):
                reference_mol = Chem.MolFromSmiles(reference_mol)
            if reference_mol:
                reference_smile = Chem.MolToSmiles(reference_mol)
                fp_ref = MACCSkeys.GenMACCSKeys(reference_mol)

                # Calculate similarities and sort
                gens = []
                for mol in generated_mols:
                    if isinstance(mol, str):
                        mol = Chem.MolFromSmiles(mol)
                    if mol:
                        fp = MACCSkeys.GenMACCSKeys(mol)
                        similarity = DataStructs.FingerprintSimilarity(fp_ref, fp, metric=DataStructs.TanimotoSimilarity)
                        gens.append((mol, similarity))

                sorted_mols = [mol for mol, _ in sorted(gens, key=lambda e: e[1], reverse=True)]
            else:
                sorted_mols = generated_mols
        else:
            sorted_mols = generated_mols

        # Convert to SMILES
        generated_smiles = []
        for mol in sorted_mols:
            if mol:
                smiles = Chem.MolToSmiles(mol) if not isinstance(mol, str) else mol
                if smiles:
                    generated_smiles.append(smiles)

        print(f"Generated {len(generated_smiles)} valid compounds")
        
        # Store in database
        print("Integrating compounds into knowledge graph...")
        stored_count = 0
        for smiles in generated_smiles:
            try:
                _key = _GenerateKey(smiles) 
                drug_id = f"drug/{_key}"
                protein_id = f"protein/{pdb_id}"

                # Skip if already exists
                if drug_collection.has(_key):
                    continue

                # Generate embedding
                embedding = GetChemBERTaEmbeddings(smiles)
                if not embedding:
                    continue
                    
                # Insert drug document
                doc = {
                    "_key": _key,
                    "_id": drug_id, 
                    "accession": "Generated",
                    "drug_name": f"Generated_Compound_{_key}",
                    "cas": "N/A",
                    "unii": "N/A",
                    "synonym": f"AI_Generated_{pdb_id}",
                    "key": _key,
                    "chembl": "N/A",
                    "smiles": smiles,
                    "inchi": "N/A",
                    "generated": True,
                    "target_pdb": pdb_id,
                    "embedding": embedding,
                    "generation_timestamp": datetime.now().isoformat()
                }
                drug_collection.insert(doc)

                # Create drug-protein link
                existing_links = list(db.aql.execute(f'''
                    FOR link IN `drug-protein` 
                    FILTER link._from == "{drug_id}" AND link._to == "{protein_id}" 
                    RETURN link
                '''))

                if not existing_links:
                    link_doc = {
                        "_from": drug_id,
                        "_to": protein_id,
                        "generated": True,
                        "generation_method": "TamGen",
                        "target_pdb": pdb_id
                    }
                    link_collection.insert(link_doc)
                    
                stored_count += 1
                
            except Exception as e:
                print(f"Error storing compound {smiles}: {str(e)}")
                continue

        # Create visual display
        valid_mols = []
        legends = []

        for i, smiles in enumerate(generated_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_mols.append(mol)
                legends.append(f"Gen-{i+1}")
            else:
                print(f"Invalid SMILES skipped: {smiles}")

        # Display compound grid
        if valid_mols:
            img = Draw.MolsToGridImage(
                valid_mols, 
                molsPerRow=3, 
                legends=legends,
                subImgSize=(200, 200)
            )
            st.write(img)

        result = {
            "reference_smile": reference_smile,
            "generated_smiles": generated_smiles,
            "compounds_stored": stored_count,
            "target_pdb": pdb_id
        }
        
        _display_sidebar_output("Generation Complete", result)

        return result

    except Exception as e:
        error_msg = f"Compound generation failed: {str(e)}"
        print(error_msg)
        _display_sidebar_output("Generation Error", error_msg, "error")
        return {"error": error_msg}
    
def FindSimilarDrugs(smiles: str, top_k: int = 6) -> List[Dict[str, Union[str, float]]]:
    """
    MOLECULAR SIMILARITY SEARCH: Discovers structurally similar drugs using high-dimensional chemical embeddings.
    
    Use this tool for:
    - Finding drug analogs and biosimilars
    - Identifying potential drug repurposing candidates
    - Analyzing chemical space and drug clusters
    - Supporting medicinal chemistry optimization
    
    Technology: ChemBERTa embeddings with cosine similarity
    Search scope: Entire drug database (approved + generated compounds)
    
    Input: SMILES string of query molecule
    Output: Ranked list of similar drugs with similarity scores

    Args:
        smiles (str): SMILES string of the query molecule
        top_k (int, optional): Number of most similar drugs to retrieve (default: 6)

    Returns:
        List[Dict[str, Union[str, float]]]: List of drug records with similarity scores
    """
    try:
        print(f"Finding drugs similar to: {smiles}")

        # Validate input
        test_mol = Chem.MolFromSmiles(smiles)
        if not test_mol:
            error_msg = f"Invalid SMILES string: {smiles}"
            _display_sidebar_output("Invalid Input", error_msg, "error")
            return []

        # Generate query embedding
        embedding = GetChemBERTaEmbeddings(smiles)
        if not embedding:
            error_msg = "Failed to generate embedding for query molecule"
            _display_sidebar_output("Embedding Error", error_msg, "error")
            return []
        
        # Execute similarity search
        aql_query = """
        LET query_vector = @query_vector
        FOR doc IN drug
            FILTER doc.embedding != null
            LET score = COSINE_SIMILARITY(doc.embedding, query_vector)
            FILTER score > 0.5
            SORT score DESC
            LIMIT @top_k
            RETURN { 
                drug_key: doc._key,
                drug_name: doc.drug_name,
                smiles: doc.smiles,
                similarity_score: score,
                generated: doc.generated,
                chembl_id: doc.chembl
            }
        """
        
        cursor = db.aql.execute(aql_query, bind_vars={
            "query_vector": embedding, 
            "top_k": top_k
        })
        results = list(cursor)

        if results:
            # Create results table
            df = pd.DataFrame(results)
            df['similarity_score'] = df['similarity_score'].round(4)
            st.table(df)
            
            _display_sidebar_output("Similarity Search", {
                "query_smiles": smiles,
                "results_found": len(results),
                "similarity_method": "ChemBERTa + Cosine Similarity",
                "min_similarity": min(r['similarity_score'] for r in results) if results else 0,
                "max_similarity": max(r['similarity_score'] for r in results) if results else 0
            })
            
        else:
            _display_sidebar_output("No Matches", f"No similar drugs found for: {smiles}", "error")
        
        return results
        
    except Exception as e:
        error_msg = f"Similarity search failed: {str(e)}"
        _display_sidebar_output("Search Error", error_msg, "error")
        return []

# ================= Enhanced Tool Wrappers =================

find_drug = Tool(
    name="FindDrug",
    func=FindDrug,
    description=FindDrug.__doc__
)

find_proteins_from_drug = Tool(
    name="FindProteinsFromDrug",
    func=FindProteinsFromDrug,
    description=FindProteinsFromDrug.__doc__
)

text_to_aql = Tool(
    name="TextToAQL",
    func=TextToAQL,
    description=TextToAQL.__doc__
)

plot_smiles_2d = Tool(
    name="PlotSmiles2D",
    func=PlotSmiles2D,
    description=PlotSmiles2D.__doc__
)

plot_smiles_3d = Tool(
    name="PlotSmiles3D",
    func=PlotSmiles3D,
    description=PlotSmiles3D.__doc__
)

predict_binding_affinity = Tool(
    name="PredictBindingAffinity",
    func=PredictBindingAffinity,
    description=PredictBindingAffinity.__doc__
)

get_amino_acid_sequence = Tool(
    name="GetAminoAcidSequence",
    func=GetAminoAcidSequence,
    description=GetAminoAcidSequence.__doc__
)

get_chemberta_embeddings = Tool(
    name="GetChemBERTaEmbeddings",
    func=GetChemBERTaEmbeddings,
    description=GetChemBERTaEmbeddings.__doc__
)

prepare_pdb_data = Tool(
    name="PreparePDBData",
    func=PreparePDBData,
    description=PreparePDBData.__doc__
)

generate_compounds = Tool(
    name="GenerateCompounds",
    func=GenerateCompounds,
    description=GenerateCompounds.__doc__
)

find_similar_drugs = Tool(
    name="FindSimilarDrugs",
    func=FindSimilarDrugs,
    description=FindSimilarDrugs.__doc__
)

# ================= Optimized Tool Collection =================

tools = [
    find_drug, 
    find_proteins_from_drug,
    text_to_aql, 
    plot_smiles_2d, 
    plot_smiles_3d,
    predict_binding_affinity,
    get_amino_acid_sequence,
    get_chemberta_embeddings,
    prepare_pdb_data,
    generate_compounds,
    find_similar_drugs
]