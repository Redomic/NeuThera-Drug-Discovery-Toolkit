<img src="https://i.imgur.com/cnpipY3.png"  width="20%" height="20%">

# NeuThera - AI-Driven Drug Discovery Toolkit

NeuThera is a modular drug discovery starter toolkit, integrating multiple state-of-the-art generative molecular design models such as [TamGen](https://www.nature.com/articles/s41467-024-53632-4), [chemBERTa](https://arxiv.org/abs/2010.09885), and [DeepPurpose](https://arxiv.org/abs/2004.08919) with other advanced tooling like graph-based retrieval, molecular fingerprint embeddings, ADMET profiling, etc. The goal is to create a fully modular agentic AI framework capable of generating, validating, and optimizing novel drug candidates in an end-to-end pipeline - creating and maintaining a graph database of all FDA approved and generated compounds for further repurposing or discovery.

NeuThera supports both **end-to-end drug discovery automation** and **interactive, user-guided drug design workflows**.

Some example generations (Protein PDB 5ool):

<img src="https://i.imgur.com/JQYUBcZ.png"  width="70%">

## Get Started

**NOTE:** We have tried to make the installation simple, but due to the usage of multiple new technologies. They might not always work well together for different systems. It's recommended that you use Linux or MacOS for now. Hopefully, in the future we can update this repo to install everything without hurdles. If you get issues with libraries or wheel building, please try installing through conda forge.

If everything still goes wrong, please feel free to message any of the contributors. Any and all contributions are also welcome to make the installation process better.

### Local Enviroment - Setting up

Requirements:

- Linux or MacOS
- Docker
- Conda

Git clone and navigate to project repo

```
git clone https://github.com/Redomic/NeuThera-Drug-Discovery-Toolkit.git
```

Setup Docker: Type in terminal - This will install and run an image of ArangoDB with password=openSesame

```
docker compose up -d
```

Create a conda environment (IMPORTANT: use version 3.12.2)

```
conda create --name neuthera python=3.12.2
conda activate neuthera
```

Install requirements

```
pip install -r requirements.txt
```

Create an .env file with the openAI key in root

```
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXX
```

Rest of the work gets done by the notebook 😎
Run the cells in **start.ipynb**

OPTIONAL: If you want to work with vector embeddings, you'll need to create ArangoDB's expiremental vector index. Run these commands after the preprocessing section of the notebook

```
docker exec -it boring_napier arangosh
```

In Arangosh

```
db._useDatabase("NeuThera");
db.drug.ensureIndex({
        name: “chemberta_cosine”
        type: “vector”
        fields: [“embedding”]
        params: { metric: “cosine”, dimension: 768, nLists: 100 }
});
```

## Architecture

NeuThera is architected as a multi-model system, integrating specialized tools and frameworks to create a starter toolkit for end-to-end drug discovery. The modular nature of the technology used allows researchers to expand the tooling, as all necessary data and functions are already provided.

1. **Core Modules**

   - **TamGen:** A transformer-based generative model for de novo molecular design, trained on 120 million+ SMILES representations of known bioactive compounds. All generated compounds are validated and docked on target protein using AutoBox before final result.
   - **DeepPurpose:** A multi-model framework used to check the binding affinity between a drug and a target protein. As all generated compounds from TamGen are already validated with docking coordinates, DeepPurpose fits in perfecting with this pipeline.
   - **chemBERTa:** Transformer-based model trained on 120+ million SMILES to create meaningful vector embeddings. Used to create vector space of all FDA Approved drugs + generated compounds for advanced analytics and repurposing.

2. **Knowledge Graph & Retrieval Modules**

   - **ArangoDB**: A multimodal database storing structured relationships between genes, proteins, diseases, drugs, and molecular structures.
   - **Graph-based querying** for retrieving relevant molecular information, linking known drug-target interactions to novel compounds.

3. **Molecular Similarity & Lead Optimization**

   - **Morgan/Circular Fingerprints:** Used for molecular embedding, leveraging Tanimoto similarity and vector representations.
   - **Vector Space Search:** for identifying structurally similar compounds within large-scale chemical libraries.
   - **Graph-based lead optimization:** Using molecular graphs and chemical property analysis to refine generated compounds.

4. **GraphRAG & Biomedical Reasoning**
   - **GraphRAG framework** Augments traditional RAG (Retrieval-Augmented Generation) methods by incorporating graph-based reasoning for biomedical data
   - **Ontological mapping**: Integrates structured biomedical ontologies (e.g., DrugBank, ChEMBL) to enhance retrieval and inference.

## Future Pipeline

- **Enhancing generative models with a feedback loop:**

  - Integrating diffusion-based generative modeling to improve the novelty, drug-likeness, and synthetic accessibility of generated compounds.
  - Implementing a reinforcement-driven feedback loop for iterative optimization
    1.  **TamGen** generates an initial batch of 10000 de novo compounds.
    2.  **DeepPurpose** predicts binding affinity for each compound with the target protein.
    3.  Compound encoders select the highest-scoring compounds as reference for next batch.
    4.  Loop continues until convergence, maximizing binding affinity while optimizing ADMET properties.

- **Automated lead optimization and drug repurposing:**

  - Integrating Graph Neural Networks (GNNs) on molecular embeddings to predict novel indications for existing compounds.
  - Scaffold hopping via vector space traversal to explore chemically diverse, high-affinity derivatives.

- **Modular tooling for user-defined extensions:**

  - Allowing users to integrate custom generative models, scoring algorithms, functions, etc.
  - Supporting custom scoring functions, including multi-objective optimization for ADMET, pharmacokinetics, and toxicity filtering.

- **Advanced UI/UX for visualization and interactive drug design:**
  - Developing an interactive web platform with:
    - AI-Chatbot system
      - Molecular visualization for ligand-protein interactions and generated compounds.
      - Graph-based relationship exploration to analyze drug-target-disease links.
      - Live compound generation and scoring dashboard, enabling real-time feedback on molecular design.
     
## Contibute

We are actively looking for contributors who are interested in this project, DM Praneeth or me (Jones) :D

[<img src="https://i.imgur.com/oz1xZLN.png"  width="10%" height="10%">](https://www.redomic.in)
