import os
import sys
import requests
import ast
import json
import hashlib
import tempfile
import re

from datetime import datetime
from glob import glob
from io import StringIO

from db import db

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from arango import ArangoClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.llms.bedrock import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

import streamlit as st
import networkx as nx
from pyvis.network import Network

import boto3

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

def FindSimilarDrugs(drug_name):
    """
    Finds the top k most similar drugs to the given drug based on cosine similarity.
    
    Args:
        drug_name (str): The name of the drug to compare.

    Returns:
        List of tuples [(drug_name, similarity_score), ...]
    """
    
    top_k = 5

    query = f"""
        FOR d IN drug
            FILTER LOWER(d.drug_name) == LOWER(@drug_name) OR LOWER(@drug_name) IN TOKENS(d.synonym, "text_en")
            RETURN d.embedding
    """
    result = list(db.aql.execute(query, bind_vars={"drug_name": drug_name}))
    
    if not result:
        raise ValueError(f"Drug '{drug_name}' not found in the database.")
    
    embedding = result[0]

    aql_query = f"""
        LET query_vector = @query_vector
        FOR d IN drug
            FILTER LOWER(d.drug_name) != LOWER(@drug_name)
            LET score = COSINE_SIMILARITY(d.embedding, query_vector)
            SORT score DESC
            LIMIT @top_k
            RETURN {{ drug: d.drug_name, similarity_score: score }}
    """

    cursor = db.aql.execute(aql_query, bind_vars={"drug_name": drug_name, "query_vector": embedding, "top_k": top_k})
    results = list(cursor)

    if results:
        df = pd.DataFrame(results)
        st.table(df)
        with st.sidebar:
            st.markdown(f"**Action Output**")
            st.code(aql_query, language="aql")
            st.json(results)
            st.divider()
        return results
    
    return results

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

def PlotSmiles2D(smiles):
    """Generates and displays a 2D molecular structure from a SMILES string.
    
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
        raise False

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

# ================= Tooling =================

find_drug_tool = Tool(
    name="FindDrug",
    func=FindDrug,
    description=FindDrug.__doc__
)

find_similar_drugs_tool = Tool(
    name="FindSimilarDrugs",
    func=FindSimilarDrugs,
    description=FindSimilarDrugs.__doc__
)

find_proteins_from_drug_tool = Tool(
    name="FindProteinsFromDrug",
    func=FindProteinsFromDrug,
    description=FindProteinsFromDrug.__doc__
)

plot_smiles_2d_tool = Tool(
    name="PlotSmiles2D",
    func=PlotSmiles2D,
    description=PlotSmiles2D.__doc__
)

plot_smiles_3d_tool = Tool(
    name="PlotSmiles3D",
    func=PlotSmiles3D,
    description=PlotSmiles3D.__doc__
)

tools = [find_drug_tool, find_similar_drugs_tool, find_proteins_from_drug_tool, plot_smiles_2d_tool, plot_smiles_3d_tool]