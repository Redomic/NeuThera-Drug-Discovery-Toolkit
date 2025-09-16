from arango import ArangoClient
from arango.exceptions import GraphListError

# 🔹 Update these
DB_NAME = "NeuThera"
USERNAME = "root"
PASSWORD = "openSesame"

# Initialize client
client = ArangoClient(hosts="http://localhost:8529")

# Connect to NeuThera DB
db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)

print(f"✅ Connected to ArangoDB database: {DB_NAME}")

# --- Test 1: list all graphs in this DB ---
try:
    graphs = db.graphs()
    print("\nGraphs in database:")
    for g in graphs:
        print(" -", g["name"])
except GraphListError as e:
    print("\n❌ Could not list graphs (GraphListError):", e)

# --- Test 2: access the NeuThera graph directly ---
try:
    graph = db.graph(DB_NAME)
    print(f"\n✅ Successfully accessed graph: {DB_NAME}")
    print("Vertex collections:", graph.vertex_collections())
    print("Edge definitions:", graph.edge_definitions())
except Exception as e:
    print(f"\n❌ Could not access graph {DB_NAME} directly:", e)
