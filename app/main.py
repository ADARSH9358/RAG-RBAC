from .rag_utils.turso_client import connect as turso_connect
import pandas as pd
import os
from pathlib import Path
from pydantic import BaseModel
import duckdb

from fastapi import FastAPI, UploadFile,File, Form, HTTPException, Depends
from fastapi import BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from passlib.hash import bcrypt
from langchain_core.documents import Document

from .rag_utils.rag_module import run_indexer,vectorstore,get_rag_chain
from .rag_utils.query_classifier import detect_query_type_llm
from .rag_utils.csv_query import ask_csv
from .rag_utils.rag_chain import ask_rag
import re

app = FastAPI()
security = HTTPBasic()
load_dotenv()


SQLITE_DB_URL = os.getenv("SQLITE_DB_PATH")
SQLITE_TOKEN  = os.getenv("SQLITE_TOKEN")



# -------------------------
# === DUCKDB SETUP ===
# -------------------------
# Use local DuckDB for CSV table storage and metadata
DUCKDB_DIR = Path("static/data")
DUCKDB_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_FILE = DUCKDB_DIR / "structured_queries.duckdb"

def init_duckdb():
    try:
        conn = duckdb.connect(str(DUCKDB_FILE))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tables_metadata (
                table_name TEXT,
                role TEXT
            )
        """)
        conn.close()
        print("✅ DuckDB initialized (local)")
    except Exception as e:
        print(f"⚠️ DuckDB init skipped due to lock: {e}")

init_duckdb()


# -------------------------
# === SQLITE DATABASE SETUP ===
# -------------------------

conn = turso_connect()
c = conn.cursor()
c.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
);

CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    role TEXT,
    filepath TEXT NOT NULL,
    headers_str TEXT,
    embedded INTEGER DEFAULT 0
);
""")
conn.commit()

def create_default_user():
    conn_local = turso_connect()
    c_local = conn_local.cursor()

    c_local.execute("INSERT OR IGNORE INTO roles (role_name) VALUES (?)", ("C-Level",))
    hashed_pw = bcrypt.hash("admin123")
    try:
        c_local.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", ("admin", hashed_pw, "C-Level"))
        conn_local.commit()
        print("✅ Default C-Level user created.")
    except Exception:
        print("⚠️ User already exists.")
    conn_local.close()


# Call it on startup
create_default_user()

# -------------------------
# === AUTHENTICATION ===
# -------------------------
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    print("username: ", username)
    print("password: ", password)
    c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    print("DB row:", row)
    if not row or not bcrypt.verify(password, row[0]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": row[1]}

# === MODELS ===
class ChatRequest(BaseModel):
    question: str

# -------------------------
# === ROUTES ===
# -------------------------
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

@app.get("/roles")
def get_roles(user=Depends(authenticate)):
    c.execute("SELECT role_name FROM roles")
    roles = [r[0] for r in c.fetchall()]
    return {"roles": roles}

@app.post("/create-user")
def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    user=Depends(authenticate)
):
    if user["role"] != "C-Level":
        raise HTTPException(status_code=403, detail="Only C-Level can create users.")

    c.execute("SELECT 1 FROM roles WHERE role_name = ?", (role,))
    if not c.fetchone():
        raise HTTPException(status_code=400, detail="Invalid role")

    hashed = bcrypt.hash(password)
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed, role))
        conn.commit()
        return {"message": f"User '{username}' added with role '{role}'"}
    except Exception:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/create-role")
def create_role(role_name: str = Form(...), user=Depends(authenticate)):
    if user["role"] != "C-Level":
        raise HTTPException(status_code=403, detail="Only C-Level can create roles.")

    try:
        c.execute("INSERT INTO roles (role_name) VALUES (?)", (role_name,))
        conn.commit()
        return {"message": f"Role '{role_name}' created"}
    except Exception:
        raise HTTPException(status_code=400, detail="Role already exists")



UPLOAD_DIR = "static/uploads"

@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...), role: str = Form(...)):
    try:
        filename = file.filename
        extension = Path(filename).suffix.lower()

        # Prepare storage
        role_dir = os.path.join(UPLOAD_DIR, role)
        os.makedirs(role_dir, exist_ok=True)
        filepath = os.path.join(role_dir, filename)

        # Read content + save file
        data = await file.read()  # Read once

        with open(filepath, "wb") as f:
            f.write(data)  # Save file for future indexing

        headers_str = None  # Default for non-CSV files

        # Convert to string content for validation (optional)
        if extension == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(data))
            content = df.to_string(index=False)

            # Load for DuckDB
            df1 = pd.read_csv(filepath)
            raw_table_name = Path(filepath).stem
            table_name = re.sub(r"[^0-9a-zA-Z_]", "_", raw_table_name)
            if not table_name or table_name[0].isdigit():
                table_name = f"table_{table_name}" if table_name else "table_uploaded"

            # Save metadata including headers
            headers = df1.columns.tolist()
            headers_str = ",".join(headers)

            # Register dataframe explicitly to avoid replacement-scan edge cases.
            duck_conn = duckdb.connect(str(DUCKDB_FILE))
            try:
                duck_conn.register("uploaded_df", df1)
                duck_conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                duck_conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM uploaded_df')
                duck_conn.unregister("uploaded_df")

                # ✅ Save metadata to local DuckDB
                duck_conn.execute(
                    "INSERT INTO tables_metadata (table_name, role) VALUES (?, ?)",
                    (table_name, role)
                )
                duck_conn.commit()
            finally:
                duck_conn.close()
            print(f"✅ CSV table '{table_name}' saved to DuckDB")

        elif extension == ".md":
            content = data.decode("utf-8")
            print(f"✅ Markdown file '{filename}' loaded")
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .md")

        # Save metadata to Turso (SQLite)
        conn = turso_connect()
        c = conn.cursor()
        c.execute("INSERT INTO documents (filename, role, filepath, headers_str, embedded) VALUES (?, ?, ?, ?, ?)",
                  (filename, role, filepath, headers_str, 0))
        conn.commit()
        conn.close()
        print(f"✅ Document metadata saved to Turso DB")
        
        # Index for RAG; quota failures should not block file upload.
        index_result = run_indexer()
        if index_result.get("quota_exceeded"):
            warning = (
                "Document uploaded, but embedding was skipped because Chroma quota is exceeded. "
                "Increase quota or clear existing vectors, then re-run indexing."
            )
            print(f"⚠️ {warning}")
            return JSONResponse(
                status_code=202,
                content={
                    "message": f"{filename} uploaded successfully for role '{role}'.",
                    "warning": warning,
                    "indexing": index_result,
                },
            )

        print("✅ Files indexed successfully")
        return JSONResponse(content={"message": f"{filename} uploaded successfully for role '{role}'."})

    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
   
"""
@app.post("/chat")
async def chat(req: ChatRequest, user=Depends(authenticate)):
    role = user["role"]
    username = user["username"]
    question = req.question

    # 1. Detect mode: SQL or RAG
    mode = detect_query_type_llm(question)
    print(mode)

    
    # 2. Route to appropriate handler
    if mode == "SQL":
        result = await ask_csv(question, role, username, return_sql=True)
        #result = await ask_csv(question) 
    else:
    
        result = await ask_rag(question, role)  # pass role to enforce role-based doc access

    return {
        "user": username,
        "role": role,
        "mode": mode,
        "answer": result["answer"],
        **({"sql": result["sql"]} if "sql" in result else {})
    }
"""
@app.post("/chat")
async def chat(req: ChatRequest, user=Depends(authenticate)):
    role = user["role"]
    username = user["username"]
    question = req.question

    # 1. Detect mode: SQL or RAG
    mode = detect_query_type_llm(question)
    print(f"Detected mode: {mode}")

    result = {}
    fallback_used = False

    # 2. Route to appropriate handler
    if mode == "SQL":
        try:
            result = await ask_csv(question, role, username, return_sql=True)

            if result.get("error") or not result.get("answer", "").strip():
                raise ValueError("SQL query blocked or failed")

        except Exception as e:
            print(f"[SQL Fallback Triggered] Error: {e}")
            result = await ask_rag(question, role)
            fallback_used = True
            mode = "SQL → fallback to RAG"

    else:
        result = await ask_rag(question, role)

    return {
        "user": username,
        "role": role,
        "mode": mode,
        "fallback": fallback_used,
        "answer": result["answer"],
        **({"sql": result["sql"]} if "sql" in result else {})
    }
