import os
import sys

import re

from typing import Any

from db import db, arango_graph

import numpy as np

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler

from rdkit.Chem import Draw, AllChem

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

import streamlit as st
from pyvis.network import Network

#================= Models & DB =================

sys.path.append(os.path.abspath("./TamGen"))

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

drug_collection = db.collection('drug')
link_collection = db.collection('drug-protein') 

# ================== Helper ==================

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

# ================= Functions =================

def FindDrug(drug_name: str):
    """
    Retrieves detailed information about a drug from the database.

    Args:
        drug_name (str): The name of the drug to search for.

    Returns:
        dict or None: A dictionary containing drug details if found, otherwise None.
    """

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
        with st.sidebar:
            st.markdown(f"**Action Output**")
            st.json(results[0])  # Use st.json for better formatting of dict output
            st.divider()
        return results[0]
    else:
        return None

def FindProteinsFromDrug(drug_name):
    """
    Finds all relevent proteins to the given drug

    Args:
        drug_name (str): The name of the drug.

    Returns:
        List[dict]: A list of PDB Ids
    """

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

    net = Network(height="500px", width="100%", directed=True, notebook=False)
    net.force_atlas_2based()

    nodes = set()
    edges = set()

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

    for doc in graph_cursor:
        if (doc["from"] != None) and (doc["to"] != None):
            from_node = doc["from"]
            to_node = doc["to"]
            edge_type = doc["type"]

            nodes.add(from_node)
            nodes.add(to_node)
            edges.add((from_node, to_node, edge_type))

    for node in nodes:
        net.add_node(node, label=node, color=color_map[classify_node(node)])

    for from_node, to_node, edge_type in edges:
        net.add_edge(from_node, to_node, title=edge_type, color="#5c677d")

    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as file:
        html = file.read()

    with st.sidebar:
        st.markdown(f"**Action Output**")
        st.code(graph_query, language="aql")
        st.json(proteins)
        st.divider()

    if proteins:
        st.components.v1.html(html, height=550, scrolling=True)

    return proteins

sanizited_schema = _SanitizeInput(d=arango_graph.schema, list_limit=32)
arango_graph.set_schema(sanizited_schema)

def TextToAQL(query: str):
    """Execute a Natural Language Query in ArangoDB, and return the result as text."""
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    class CallbackHandler(BaseCallbackHandler):
        def on_agent_action(self, action, **kwargs):
            thought = action.log.split('\n')[0].replace('Thought:', '').strip()
            print(thought)
            step = {
                'type': 'GraphRAG Agent',
                'content': f"🤔 **Thought:** {thought}",
                # 'tool': action.tool,
                # 'input': action.tool_input
            }
            
            with st.sidebar:
                st.markdown(f"**Thought**")
                st.markdown(step['content'])
                # st.markdown(f"🔧 **Action:** {step['tool']}")
                # st.markdown(f"📤 **Input:** `{step['input']}`")
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

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True,
        callbacks=[CallbackHandler()]
    )
    
    result = chain.invoke(query)

    return str(result["result"])

def PlotSmiles2D(smiles):
    """Generates and displays a 2D molecular structure from a SMILES string. If you have multiple compounds, use this function one by one for each string
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        Boolean - If plotted or not
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.write(Draw.MolToImage(mol, size=(300, 300))) 
        return True
    else:
        return False

def PlotSmiles3D(smiles):
    """Generates an interactive 3D molecular structure from a SMILES string.
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        boolean for if plotted or not 
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol) 

    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status == -1: 
        return False

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return False

    conformer = mol.GetConformer()
    if not conformer.Is3D():
        return False

    atom_positions = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    atom_colors = ['green' if atom == 'O' else 'red' if atom == 'H' else 'blue' for atom in atom_symbols]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=atom_positions[:, 0], y=atom_positions[:, 1], z=atom_positions[:, 2],
        mode='markers+text',
        marker=dict(size=3, color=atom_colors, opacity=0.8),
        text=atom_symbols,
        textposition="top center",
        showlegend=False 
    ))

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fig.add_trace(go.Scatter3d(
            x=[atom_positions[start][0], atom_positions[end][0]],
            y=[atom_positions[start][1], atom_positions[end][1]],
            z=[atom_positions[start][2], atom_positions[end][2]],
            mode='lines',
            line=dict(color='gray', width=3),
            showlegend=False 
        ))

    fig.update_layout(
        title="3D Molecular Structure",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        width=600, height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    if fig:
        with st.sidebar:
            st.markdown(f"**2D - TO - 3D**")
            st.write(Draw.MolToImage(mol, size=(300, 300)))
            st.divider()
        st.write(fig)
        return True
    else:
        return False    

# ================= Tooling Wrapper =================

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

tools = [
    find_drug, 
    text_to_aql, 
    find_proteins_from_drug, 
    plot_smiles_2d, 
    plot_smiles_3d
]   