import re
import duckdb
import os
import tabulate
from openai import OpenAI
from .turso_client import connect as turso_connect
from pathlib import Path


DUCKDB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "static", "data", "structured_queries.duckdb")
os.makedirs(os.path.dirname(DUCKDB_FILE), exist_ok=True)

if not os.path.exists(DUCKDB_FILE):
    duckdb.connect(DUCKDB_FILE).close()

duck_conn = None

def get_duck_conn():
    global duck_conn
    if duck_conn is not None:
        return duck_conn

    # Use a single consistent connection mode across the app process.
    # Mixing read_only and read_write connections for the same DB file raises
    # "different configuration than existing connections" in DuckDB.
    duck_conn = duckdb.connect(DUCKDB_FILE)
    print("✅ DuckDB initialized")

    return duck_conn


# OpenAI setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_allowed_tables_for_role(role: str) -> list[str]:
    conn = get_duck_conn()
    if role.lower() == "c-level":
        query = "SELECT table_name FROM tables_metadata"
        return [row[0] for row in conn.execute(query).fetchall()]
    elif role.lower() == "general":
        query = "SELECT table_name FROM tables_metadata WHERE role = 'general'"
        return [row[0] for row in conn.execute(query).fetchall()]
    else:
        query = """
        SELECT table_name FROM tables_metadata
        WHERE role = ? OR role = 'general'
        """
        return [row[0] for row in conn.execute(query, [role]).fetchall()]

def extract_tables_from_sql(sql: str) -> list[str]:
    # Extract tables used in FROM and JOIN clauses
    return re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, flags=re.IGNORECASE)

def flatten_matches(matches: list[tuple]) -> list[str]:
    return [item for tup in matches for item in tup if item]

FORBIDDEN = ["insert", "update", "delete", "drop", "alter", "create"]

def is_safe_query(sql: str) -> bool:
    lowered = sql.strip().lower().rstrip(";")
    return lowered.startswith("select") and all(word not in lowered for word in FORBIDDEN)

def translate_nl_to_sql(question: str, allowed_tables: list[str]) -> str:
    print("translate_nl_to_sql() called")
    conn = turso_connect()
    print("Connecting to Turso deployed DB")
    cur = conn.cursor()

    # fetch headers from table
    cur.execute("""
        SELECT filename, headers_str FROM documents 
        WHERE embedded = 1 AND headers_str IS NOT NULL
    """)
    rows = cur.fetchall()
    print("Raw rows from DB:", rows)
    conn.close()

    schemas = []
    for filename, headers_str in rows:
        try:
            print("inside schemas")
            raw_table_name = Path(filename).stem
            table_name = re.sub(r"[^0-9a-zA-Z_]", "_", raw_table_name)
            if not table_name or table_name[0].isdigit():
                table_name = f"table_{table_name}" if table_name else "table_uploaded"
            print(table_name)
            cols = ", ".join(headers_str.split(","))
            print(cols)
            schemas.append(f"Table: {table_name}\nColumns: {cols}")
        except Exception as e:
            print(f"❌ Error while building schema for {filename}: {e}")


    print("Schemas:", schemas)

    schema_block = "\n\n".join(schemas)
    print("schema_block:\n", schema_block)

    # Prompt for LLM
    prompt = f"""
    You are an assistant that converts natural language questions into safe SQL SELECT queries.

    Use only the following schemas:
    {schema_block}

    Constraints:
    - Use only the tables listed above.
    - Use the exact column names as-is (including hyphens, underscores, casing).
    - Return only a SELECT query (no INSERT/UPDATE/DELETE).
    - If asked about 'employee name', consider alternatives like 'full-name', 'last-name'.
    - If asked about 'position', consider synonyms like 'role', 'designation'.
    - Do not mix aggregate functions (like COUNT(*)) with *. Use either a grouped summary or return them separately."
    Natural Language Question: "{question}"

    SQL:
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        print("LLM call successful")
        
        response_text = response.choices[0].message.content.strip()
        
        print("Raw SQL from LLM:\n", response_text)

        return response_text
        #return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ LLM call failed:", e)
        return "Error generating SQL"

#async def ask_csv(question: str, role: str) -> dict:
async def ask_csv(question: str, role: str, username: str, return_sql: bool = False) -> dict:
    conn = get_duck_conn()
    allowed_tables = get_allowed_tables_for_role(role)

    try:
        sql = translate_nl_to_sql(question, allowed_tables)
        print(f"[SQL GENERATED]:\n{sql}")

        if not is_safe_query(sql):
            return {"answer": "Only SELECT queries are allowed.", "error": True}

        raw_matches = extract_tables_from_sql(sql)
        referenced_tables = flatten_matches(raw_matches)

        for table in referenced_tables:
            if table not in allowed_tables:
                return {"answer": f"Access denied to table: {table}", "error": True}

        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]
        output = [list(row) for row in result]

        markdown_table = tabulate.tabulate(output, headers=columns, tablefmt="github")
        response = {
            "answer": markdown_table if output else "Query executed, but no results found."
        }

        if return_sql:
            response["sql"] = sql

        return response

    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "error": True}
