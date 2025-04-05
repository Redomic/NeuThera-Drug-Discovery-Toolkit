import os
import sys

# import requests
# import ast
# import json
# import hashlib
# import tempfile
import re

from typing import Any
# from datetime import datetime
# from glob import glob
# from io import StringIO

from db import db, arango_graph

# import pandas as pd
import numpy as np

from dotenv import load_dotenv
# from arango import ArangoClient

# from transformers import AutoTokenizer, AutoModel
# import torch

# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
# from langchain.llms.bedrock import Bedrock
# from langchain_community.graphs import ArangoGraph
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler

# from pydantic import BaseModel, Field

# from Bio.PDB import MMCIFParser

# from rdkit import Chem, DataStructs
# from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

import streamlit as st
# import networkx as nx
from pyvis.network import Network

# from DeepPurpose import utils
# from DeepPurpose import DTI as models

# from TamGen_custom import TamGenCustom

#================= Models & DB =================

sys.path.append(os.path.abspath("./TamGen"))

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

drug_collection = db.collection('drug')
link_collection = db.collection('drug-protein') 

# tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
# model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# worker = TamGenCustom(
#     data="./TamGen_Demo_Data",
#     ckpt="checkpoints/crossdock_pdb_A10/checkpoint_best.pt",
#     use_conditional=True
# )

# ================== Helper ==================

# def _GenerateKey(smiles):
#     """Generate a unique _key for the compound using SMILES hash."""
#     hash_value = hashlib.sha256(smiles.encode()).hexdigest()[:8]
#     return f"GEN:{hash_value}"

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

# def PredictBindingAffinity(input_data, y=[7.635]):
#     """
#     Predicts the binding affinity for given drug and target sequences.

#     Parameters:
#     input_data (dict): Dictionary containing:
#         - x_drug (str): SMILES representation of the drug.
#         - x_target (str): Amino acid sequence of the protein target.

#     Returns:
#     float: Predicted binding affinity (log(Kd) or log(Ki)).
#     """

#     if isinstance(input_data, str): 
#         input_data = json.loads(input_data)

#     x_drug = input_data.get("x_drug")
#     x_target = input_data.get("x_target")

#     if not x_drug or not x_target:
#         raise ValueError("Both x_drug and x_target must be provided in input_data")

#     print("Predicting binding affinity: ", x_drug, x_target)

#     X_drug = [x_drug]
#     X_target = [x_target]
    
#     model = models.model_pretrained(path_dir='DTI_model')
#     X_pred = utils.data_process(X_drug, X_target, y, drug_encoding='CNN', target_encoding='CNN', split_method='no_split')
#     predictions = model.predict(X_pred)

#     print(predictions[0])

#     with st.sidebar:
#         st.markdown(f"**Action Output**")
#         st.markdown(
#             f":green-badge[:material/check_circle: Success] Binding Affinity: {str(predictions[0])}"
#         )
#         st.divider()

#     return predictions[0]

# def GetAminoAcidSequence(pdb_id):    
#     """
#     Extracts amino acid sequences from a given PDB structure file in CIF format.

#     DO NOT OUTPUT TO THE USER THE RESULT OF THIS FUNCTION.

#     If all the user asked for is the amino acid sequence, then say that you successfully extracted it

#     Args:
#         pdb_id (str): pdb id of the protein.

#     Returns:
#         dict: A dictionary where keys are chain IDs and values are amino acid sequences.
#     """

#     print("Getting Amino Acid sequence for ", pdb_id)

#     cif_file_path = f"./database/PDBlib/{pdb_id.lower()}.cif"

#     parser = MMCIFParser(QUIET=True)
#     structure = parser.get_structure("protein", cif_file_path)
    
#     sequences = {}
#     for model in structure:
#         for chain in model:
#             seq = "".join(residue.resname for residue in chain if residue.id[0] == " ")
#             sequences[chain.id] = seq 
            
#     with st.sidebar:
#         st.markdown(f"**Action Output**")
#         st.markdown(
#             f":green-badge[:material/check_circle: Success] Sequences prepared for {pdb_id}"
#         )
#         st.divider()
    
#     return sequences

# def GetChemBERTaEmbeddings(smiles):
#     """
#     Generate a ChemBERTa vector embedding for a given molecule represented as a SMILES string.

#     Args:
#         smiles (str): A valid SMILES representation of a molecule.

#     Returns:
#         List[float] or None: A 768-dimensional vector as a list of floats if successful, 
#                              otherwise None if the input is invalid.
#     """
    
#     print("Getting vector embedding")

#     if not isinstance(smiles, str) or not smiles.strip():
#         return None 

#     inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     with st.sidebar:
#         st.markdown(f"**Action Output**")
#         st.badge("Success", icon=":material/check:", color="green")
#         st.divider()

#     return outputs.last_hidden_state.mean(dim=1).tolist()[0]

# def PreparePDBData(pdb_id):
#     """
#     Checks if the PDB data for the given PDB ID is available.  
#     If not, downloads and processes the data.

#     ALWAYS RUN THIS FUNCTION BEFORE WORKING WITH PDB

#     Args:
#         pdb_id (str): PDB ID of the target structure.

#     """

#     DemoDataFolder="TamGen_Demo_Data"
#     ligand_inchi=None
#     thr=10

#     out_split = pdb_id.lower()
#     FF = glob(f"{DemoDataFolder}/*")
#     for ff in FF:
#         if f"gen_{out_split}" in ff:
#             print(f"{pdb_id} is downloaded")
#             return
    
#     os.makedirs(DemoDataFolder, exist_ok=True)
    
#     with open("tmp_pdb.csv", "w") as fw:
#         if ligand_inchi is None:
#             print("pdb_id", file=fw)
#             print(f"{pdb_id}", file=fw)
#         else:
#             print("pdb_id,ligand_inchi", file=fw)
#             print(f"{pdb_id},{ligand_inchi}", file=fw)
    
#     with st.sidebar:
#         st.markdown(f"**Action Output**")
#         st.markdown(
#             f":green-badge[:material/check_circle: Success] PDB Data prepared for {pdb_id}"
#         )
#         st.divider()

#     script_path = os.path.abspath("TamGen/scripts/build_data/prepare_pdb_ids.py")
#     os.system(f"python {script_path} tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
#     os.remove("tmp_pdb.csv")

# def GenerateCompounds(pdb_id):
#     """
#     Generates and sorts compounds based on similarity to a reference molecule, 
#     all generated compounds are added back to the database for futher inference.

#     Parameters:
#     - pdb_id (str): The PDB ID of the target protein.

#     Returns:
#     - dict: {
#         'reference_smile': SMILE string of the reference compound
#         'generated_smiles': [list of SMILES strings, sorted by similarity to reference]
#       }
#     """

#     num_samples=3
#     max_seed=5

#     print("Generating Compounds for PDB ", pdb_id)
#     try:
#         worker.reload_data(subset=f"gen_{pdb_id.lower()}")

#         print(f"Generating {num_samples} compounds...")

#         with st.sidebar:
#             st.markdown(
#                 f"Generating {num_samples} compounds..."
#             )
#             st.divider()

#         generated_mols, reference_mol = worker.sample(
#             m_sample=num_samples, 
#             maxseed=max_seed
#         )

#         if reference_mol:
#             if isinstance(reference_mol, str):
#                 reference_mol = Chem.MolFromSmiles(reference_mol)

#             fp_ref = MACCSkeys.GenMACCSKeys(reference_mol)

#             gens = []
#             for mol in generated_mols:
#                 if isinstance(mol, str):
#                     mol = Chem.MolFromSmiles(mol)
#                 if mol:
#                     fp = MACCSkeys.GenMACCSKeys(mol)
#                     similarity = DataStructs.FingerprintSimilarity(fp_ref, fp, metric=DataStructs.TanimotoSimilarity)
#                     gens.append((mol, similarity))

#             sorted_mols = [mol for mol, _ in sorted(gens, key=lambda e: e[1], reverse=True)]
        
#         else:
#             sorted_mols = generated_mols

#         generated_smiles = [Chem.MolToSmiles(mol) for mol in sorted_mols if mol]

#         reference_smile = Chem.MolToSmiles(reference_mol)

#         print("Generated smiles:", generated_smiles)
        
#         print("Inserting to ArangoDB...")
#         for smiles in generated_smiles:
#             _key = _GenerateKey(smiles) 
#             drug_id = f"drug/{_key}"
#             protein_id = f"protein/{pdb_id}"

#             if drug_collection.has(_key):
#                 continue

#             embedding = GetChemBERTaEmbeddings(smiles)
#             doc = {
#                 "_key": _key,
#                 "_id": drug_id, 
#                 "accession": "NaN",
#                 "drug_name": "NaN",
#                 "cas": "NaN",
#                 "unii": "NaN",
#                 "synonym": "NaN",
#                 "key": "NaN",
#                 "chembl": "NaN",
#                 "smiles": smiles,
#                 "inchi": "NaN",
#                 "generated": True,
#                 "embedding": embedding
#             }
#             drug_collection.insert(doc)

#             existing_links = list(db.aql.execute(f'''
#                 FOR link IN `drug-protein` 
#                 FILTER link._from == "{drug_id}" AND link._to == "{protein_id}" 
#                 RETURN link
#             '''))

#             if not existing_links:
#                 link_doc = {
#                     "_from": drug_id,
#                     "_to": protein_id,
#                     "generated": True
#                 }
#                 link_collection.insert(link_doc)

#         valid_mols = []
#         legends = []

#         for i, smiles in enumerate(generated_smiles):
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 valid_mols.append(mol)
#                 legends.append(f"gen={i}")
#             else:
#                 print(f"Invalid SMILES skipped: {smiles}")

#         if valid_mols:
#             img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, legends=legends)
#             st.write(img)

#         with st.sidebar:
#             st.markdown(f"**Action Output**")
#             st.json({
#                 "reference_smile": reference_smile,
#                 "generated_smiles": generated_smiles
#             })
#             st.divider()

#         return {
#             "reference_smile": reference_smile,
#             "generated_smiles": generated_smiles
#         }

#     except Exception as e:
#         print(f"Error in compound generation: {str(e)}")
#         return {"error": str(e)}
    
# def FindSimilarDrugs(smile, top_k=6):
#     """
#     Finds the top K most similar drugs based on given smile of a query molecule. Automatically gets vector embeddings.

#     Args:
#         smile (string): Smile of the query molecule.
#         top_k (int, optional): Number of most similar drugs to retrieve. Default is 5.

#     Returns:
#         List[Dict{str, [float]}]: A list of (drug_name, similarity_score) sorted by similarity.
#     """
    
#     print("Finding similar drugs...")

#     embedding = GetChemBERTaEmbeddings(smile)
    
#     aql_query = f"""
#     LET query_vector = @query_vector
#     FOR doc IN drug
#         LET score = COSINE_SIMILARITY(doc.embedding, query_vector)
#         SORT score DESC
#         LIMIT @top_k
#         RETURN {{ drug: doc._key, similarity_score: score }}
#     """
    
#     cursor = db.aql.execute(aql_query, bind_vars={"query_vector": embedding, "top_k": top_k})
#     results = list(cursor)

#     if results:
#         df = pd.DataFrame(results)
#         st.table(df)
#         with st.sidebar:
#             st.markdown(f"**Action Output**")
#             st.code(aql_query, language="aql")
#             st.json(results)
#             st.divider()
#         return results
    
#     return results

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

# predict_binding_affinity = Tool(
#     name="PredictBindingAffinity",
#     func=PredictBindingAffinity,
#     description=PredictBindingAffinity.__doc__
# )

# get_amino_acid_sequence = Tool(
#     name="GetAminoAcidSequence",
#     func=GetAminoAcidSequence,
#     description=GetAminoAcidSequence.__doc__
# )

# get_chemberta_embeddings = Tool(
#     name="GetChemBERTaEmbeddings",
#     func=GetChemBERTaEmbeddings,
#     description=GetChemBERTaEmbeddings.__doc__
# )

# prepare_pdb_data = Tool(
#     name="PreparePDBData",
#     func=PreparePDBData,
#     description=PreparePDBData.__doc__
# )

# generate_compounds = Tool(
#     name="GenerateCompounds",
#     func=GenerateCompounds,
#     description=GenerateCompounds.__doc__
# )

# find_similar_drugs = Tool(
#     name="FindSimilarDrugs",
#     func=FindSimilarDrugs,
#     description=FindSimilarDrugs.__doc__
# )

tools = [
    find_drug, 
    # find_similar_drugs, 
    text_to_aql, 
    find_proteins_from_drug, 
    plot_smiles_2d, 
    plot_smiles_3d,
    # predict_binding_affinity,
    # get_amino_acid_sequence,
    # get_chemberta_embeddings,
    # prepare_pdb_data,
    # generate_compounds,
]   