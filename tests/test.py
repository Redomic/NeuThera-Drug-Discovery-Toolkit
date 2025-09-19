
import pytest
from tools import tools

# Utility to fetch tool function by name
def get_tool(name: str):
    for t in tools:
        if t.name == name:
            return t.func
    raise ValueError(f"Tool {name} not found")

def test_find_drug():
    fn = get_tool("find_drug")
    result = fn("aspirin")
    assert result is not None
    assert "aspirin" in str(result).lower()

def test_find_proteins_from_drug():
    fn = get_tool("find_proteins_from_drug")
    proteins = fn("aspirin")
    assert isinstance(proteins, list)

def test_text_to_aql():
    fn = get_tool("text_to_aql")
    query = fn("Find drugs targeting TP53")
    assert "FOR" in query or "RETURN" in query

def test_plot_smiles_2d(tmp_path):
    fn = get_tool("plot_smiles_2d")
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    file = tmp_path / "aspirin.png"
    fn(smiles, str(file))
    assert file.exists()

def test_plot_smiles_3d():
    fn = get_tool("plot_smiles_3d")
    smiles = "CCO"
    result = fn(smiles)
    assert result is not None

def test_predict_binding_affinity():
    fn = get_tool("predict_binding_affinity")
    score = fn("CCO", "MGLSDGEWQLVLNVWGK")  # ethanol vs hemoglobin peptide
    assert isinstance(score, float)

def test_get_amino_acid_sequence():
    fn = get_tool("get_amino_acid_sequence")
    seq = fn("tests/sample_data/pdb1crn.ent")
    assert isinstance(seq, str)
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq)

def test_get_chemberta_embeddings():
    fn = get_tool("get_chemberta_embeddings")
    emb = fn("CCO")
    assert isinstance(emb, list)
    assert all(isinstance(x, float) for x in emb)

def test_prepare_pdb_data():
    fn = get_tool("prepare_pdb_data")
    result = fn("tests/sample_data/pdb1crn.ent")
    assert "atoms" in result

def test_generate_compounds():
    fn = get_tool("generate_compounds")
    result = fn("anti-cancer")
    assert isinstance(result, list)

def test_find_similar_drugs():
    fn = get_tool("find_similar_drugs")
    result = fn("aspirin")
    assert isinstance(result, list)

def test_analyse_proteins():
    fn = get_tool("analyse_proteins")
    result = fn("tests/sample_data/pdb1crn.ent")
    assert isinstance(result, dict)

def test_predict_admet_properties():
    fn = get_tool("predict_admet_properties")
    result = fn("CCO")
    assert isinstance(result, dict)
    assert "toxicity" in result

def test_predict_protein_disorder_regions_from_pdb():
    fn = get_tool("predict_protein_disorder_regions_from_pdb")
    result = fn("tests/sample_data/pdb1crn.ent")
    assert isinstance(result, list)

def test_protein_conservation_from_pdb():
    fn = get_tool("protein_conservation_from_pdb")
    result = fn("tests/sample_data/pdb1crn.ent")
    assert isinstance(result, dict)

def test_predict_drug_drug_interactions():
    fn = get_tool("predict_drug_drug_interactions")
    score = fn("CC(=O)Oc1ccccc1C(=O)O", "NC(=O)C1=CN([C@@H]2OC@HC@@H[C@H]2O)C=CC1")
    assert isinstance(score, float)
