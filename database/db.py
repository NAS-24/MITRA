import os
import json
import uuid
from datetime import datetime
from typing import Optional
 
import numpy as np
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
 
load_dotenv()
 
# ─────────────────────────────────────────────
# CONFIG  (reads from .env, falls back to defaults)
# ─────────────────────────────────────────────
_CFG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME",     "mitra_db"),
    "charset":  "utf8mb4",
    "autocommit": False,
}
 
 
# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def _connect() -> mysql.connector.MySQLConnection:
    """Returns an open connection. Raises on failure."""
    return mysql.connector.connect(**_CFG)
 
 
# ─────────────────────────────────────────────
# DATABASE & TABLE SETUP
# ─────────────────────────────────────────────
def setup(sql_file: str = None):
    """
    Creates the mitra_db database and all tables.
    Call once on first run, or whenever you need to reset.
 
    If sql_file is given, runs that SQL file (mitra_schema.sql).
    Otherwise, runs the embedded minimal schema below — useful
    when the SQL file isn't in the working directory.
    """
    # Step 1: create DB if missing
    cfg_no_db = {k: v for k, v in _CFG.items() if k != "database"}
    cfg_no_db.pop("autocommit", None)
    try:
        conn = mysql.connector.connect(**cfg_no_db)
        cur  = conn.cursor()
        cur.execute(
            "CREATE DATABASE IF NOT EXISTS mitra_db "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        conn.commit()
        cur.close(); conn.close()
        print("[DB] Database 'mitra_db' ready.")
    except Error as e:
        print(f"[DB ERROR] Could not create database: {e}")
        return False
 
    # Step 2: create tables
    conn = _connect()
    cur  = conn.cursor()
    try:
        if sql_file and os.path.exists(sql_file):
            with open(sql_file, "r", encoding="utf-8") as f:
                sql = f.read()
            stmts = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in stmts:
                if stmt.upper().startswith(("--", "/*", "USE ", "CREATE DATABASE")):
                    continue
                try:
                    cur.execute(stmt)
                except Error as e:
                    if e.errno in (1050, 1061):   # table/index already exists
                        continue
                    print(f"[DB WARN] {e}")
        else:
            # Inline minimal schema
            _run_inline_schema(cur)
 
        conn.commit()
        print("[DB] All tables ready.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] setup() failed: {e}")
        return False
    finally:
        cur.close(); conn.close()
 
 
def _run_inline_schema(cur):
    """Minimal inline schema — runs when mitra_schema.sql is absent."""
    statements = [
        """CREATE TABLE IF NOT EXISTS caregivers (
            caregiver_id  INT          AUTO_INCREMENT PRIMARY KEY,
            full_name     VARCHAR(100) NOT NULL,
            email         VARCHAR(150) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            phone         VARCHAR(20),
            created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
            last_login    DATETIME,
            is_active     BOOLEAN      DEFAULT TRUE
        )""",
        """CREATE TABLE IF NOT EXISTS patients (
            patient_id   INT          AUTO_INCREMENT PRIMARY KEY,
            caregiver_id INT          NOT NULL,
            full_name    VARCHAR(100) NOT NULL,
            date_of_birth DATE,
            notes        TEXT,
            created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
            is_active    BOOLEAN      DEFAULT TRUE,
            FOREIGN KEY (caregiver_id) REFERENCES caregivers(caregiver_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS known_persons (
            person_id       INT          AUTO_INCREMENT PRIMARY KEY,
            patient_id      INT          NOT NULL,
            person_uuid     VARCHAR(36)  NOT NULL UNIQUE,
            full_name       VARCHAR(100) NOT NULL,
            relationship    VARCHAR(100),
            photo_path      VARCHAR(500),
            caregiver_notes TEXT,
            added_by        INT,
            last_met        DATETIME,
            created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
            updated_at      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active       BOOLEAN      DEFAULT TRUE,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS face_embeddings (
            embedding_id   INT          AUTO_INCREMENT PRIMARY KEY,
            person_id      INT          NOT NULL,
            embedding_data JSON         NOT NULL,
            model_used     VARCHAR(100) DEFAULT 'ArcFace',
            photo_path     VARCHAR(500),
            created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES known_persons(person_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS interaction_timeline (
            interaction_id     INT         AUTO_INCREMENT PRIMARY KEY,
            patient_id         INT         NOT NULL,
            person_id          INT,
            recognition_status ENUM('recognized','unknown') NOT NULL,
            confidence_score   FLOAT,
            snapshot_path      VARCHAR(500),
            device_info        VARCHAR(200),
            interaction_at     DATETIME    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (person_id)  REFERENCES known_persons(person_id)
                ON DELETE SET NULL ON UPDATE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS unknown_face_queue (
            queue_id       INT          AUTO_INCREMENT PRIMARY KEY,
            patient_id     INT          NOT NULL,
            snapshot_path  VARCHAR(500),
            embedding_data JSON,
            status         ENUM('pending','accepted','rejected') DEFAULT 'pending',
            reviewed_by    INT,
            reviewed_at    DATETIME,
            created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        )""",
    ]
    for stmt in statements:
        cur.execute(stmt)
 
 
# ─────────────────────────────────────────────
# MATCHER-COMPATIBLE API
# These replace the JSON file helpers in matcher.py
# ─────────────────────────────────────────────
 
def load_db(patient_id: int = 1) -> dict:
    """
    Loads all known persons + their embeddings for a given patient.
 
    Returns a dict in the exact same format that matcher.py expects:
    {
        "Full Name": {
            "id":           "uuid-string",
            "embedding":    np.array([...]),
            "relationship": "Daughter",
            "notes":        "...",
            "last_met":     "2026-04-19T10:30:00"
        },
        ...
    }
    """
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT kp.person_uuid  AS id,
                   kp.full_name,
                   kp.relationship,
                   kp.caregiver_notes AS notes,
                   kp.last_met,
                   fe.embedding_data
            FROM   known_persons  kp
            JOIN   face_embeddings fe ON fe.person_id = kp.person_id
            WHERE  kp.patient_id = %s
              AND  kp.is_active  = TRUE
        """, (patient_id,))
        rows = cur.fetchall()
    finally:
        cur.close(); conn.close()
 
    db = {}
    for row in rows:
        name = row["full_name"]
        emb  = np.array(json.loads(row["embedding_data"]))
        # Latest embedding wins if multiple exist for same name
        db[name] = {
            "id":           row["id"],
            "embedding":    emb,
            "relationship": row["relationship"] or "Unknown",
            "notes":        row["notes"]         or "",
            "last_met":     row["last_met"].isoformat() if row["last_met"] else "",
        }
    return db
 
 
def save_db(db: dict, patient_id: int = 1) -> None:
    """
    Persists the in-memory db dict back to MySQL.
    Matches the save_db(db) signature used in pipeline.py.
 
    Only updates `last_met` for existing persons.
    New persons (not yet in DB) are inserted with their embedding.
    """
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        for name, data in db.items():
            person_uuid = data.get("id") or str(uuid.uuid4())
 
            # Check if this person already exists
            cur.execute(
                "SELECT person_id FROM known_persons WHERE person_uuid = %s",
                (person_uuid,)
            )
            row = cur.fetchone()
 
            if row:
                # Update last_met only
                cur.execute(
                    "UPDATE known_persons SET last_met = %s WHERE person_uuid = %s",
                    (data.get("last_met"), person_uuid)
                )
            else:
                # Insert new person
                cur.execute("""
                    INSERT INTO known_persons
                        (patient_id, person_uuid, full_name, relationship, caregiver_notes,
                         last_met, added_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    patient_id,
                    person_uuid,
                    name,
                    data.get("relationship", "Unknown"),
                    data.get("notes", ""),
                    data.get("last_met"),
                    None,          # added_by caregiver — set externally if needed
                ))
                new_person_id = cur.lastrowid
 
                # Save embedding
                embedding_list = (
                    data["embedding"].tolist()
                    if isinstance(data["embedding"], np.ndarray)
                    else list(data["embedding"])
                )
                cur.execute("""
                    INSERT INTO face_embeddings (person_id, embedding_data, model_used)
                    VALUES (%s, %s, %s)
                """, (new_person_id, json.dumps(embedding_list), "ArcFace"))
 
        conn.commit()
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] save_db(): {e}")
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# PERSON MANAGEMENT (Caregiver Panel)
# ─────────────────────────────────────────────
 
def register_person(patient_id: int, name: str, relationship: str,
                    notes: str, embedding: np.ndarray,
                    caregiver_id: int = None,
                    photo_path: str = None) -> str:
    """
    Registers a completely new person — creates known_persons row +
    face_embeddings row in one transaction.
 
    Returns the new person_uuid (matches pipeline.py's person_id usage).
    """
    person_uuid = str(uuid.uuid4())
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO known_persons
                (patient_id, person_uuid, full_name, relationship,
                 caregiver_notes, photo_path, added_by, last_met)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            patient_id, person_uuid, name, relationship,
            notes, photo_path, caregiver_id,
            datetime.now().isoformat()
        ))
        person_id = cur.lastrowid
 
        emb_list = (
            embedding.tolist()
            if isinstance(embedding, np.ndarray)
            else list(embedding)
        )
        cur.execute("""
            INSERT INTO face_embeddings (person_id, embedding_data, model_used, photo_path)
            VALUES (%s, %s, %s, %s)
        """, (person_id, json.dumps(emb_list), "ArcFace", photo_path))
 
        conn.commit()
        print(f"[DB] Registered '{name}' → {person_uuid}")
        return person_uuid
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] register_person(): {e}")
        return None
    finally:
        cur.close(); conn.close()
 
 
def update_person(person_uuid: str, name: str = None,
                  relationship: str = None, notes: str = None) -> bool:
    """Updates mutable fields on a known person (Caregiver Panel use)."""
    updates, values = [], []
    if name:         updates.append("full_name = %s");       values.append(name)
    if relationship: updates.append("relationship = %s");    values.append(relationship)
    if notes:        updates.append("caregiver_notes = %s"); values.append(notes)
    if not updates:
        return False
 
    values.append(person_uuid)
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute(
            f"UPDATE known_persons SET {', '.join(updates)} WHERE person_uuid = %s",
            tuple(values)
        )
        conn.commit()
        return cur.rowcount > 0
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] update_person(): {e}")
        return False
    finally:
        cur.close(); conn.close()
 
 
def deactivate_person(person_uuid: str) -> bool:
    """Soft-deletes a known person (sets is_active = FALSE)."""
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE known_persons SET is_active = FALSE WHERE person_uuid = %s",
            (person_uuid,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        cur.close(); conn.close()
 
 
def get_all_persons(patient_id: int) -> list:
    """Returns all active known persons for a patient (Caregiver Panel)."""
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT person_uuid AS id, full_name, relationship,
                   caregiver_notes AS notes, photo_path, last_met, created_at
            FROM   known_persons
            WHERE  patient_id = %s AND is_active = TRUE
            ORDER  BY full_name
        """, (patient_id,))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# CAREGIVER MANAGEMENT
# ─────────────────────────────────────────────
 
def add_caregiver(full_name: str, email: str,
                  password_hash: str, phone: str = None) -> Optional[int]:
    """
    Adds a caregiver account.
    password_hash: pass in a bcrypt hash — never store plaintext.
    Returns new caregiver_id.
    """
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO caregivers (full_name, email, password_hash, phone)
            VALUES (%s, %s, %s, %s)
        """, (full_name, email, password_hash, phone))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] add_caregiver(): {e}")
        return None
    finally:
        cur.close(); conn.close()
 
 
def get_caregiver_by_email(email: str) -> Optional[dict]:
    """Fetches a caregiver row by email (for login)."""
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT * FROM caregivers WHERE email = %s AND is_active = TRUE",
            (email,)
        )
        return cur.fetchone()
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# PATIENT MANAGEMENT
# ─────────────────────────────────────────────
 
def add_patient(caregiver_id: int, full_name: str,
                date_of_birth: str = None, notes: str = None) -> Optional[int]:
    """Adds a patient. Returns new patient_id."""
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patients (caregiver_id, full_name, date_of_birth, notes)
            VALUES (%s, %s, %s, %s)
        """, (caregiver_id, full_name, date_of_birth, notes))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] add_patient(): {e}")
        return None
    finally:
        cur.close(); conn.close()
 
 
def get_patients(caregiver_id: int) -> list:
    """Returns all active patients for a caregiver."""
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT patient_id, full_name, date_of_birth, notes, created_at
            FROM   patients
            WHERE  caregiver_id = %s AND is_active = TRUE
        """, (caregiver_id,))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# INTERACTION TIMELINE
# Called from gen_ai/api.py after each recognition
# ─────────────────────────────────────────────
 
def log_interaction(patient_id: int,
                    recognition_status: str,
                    person_uuid: str = None,
                    confidence_score: float = None,
                    snapshot_path: str = None,
                    device_info: str = None) -> Optional[int]:
    """
    Logs a recognition event. Resolves person_uuid → person_id internally.
    Returns new interaction_id.
    """
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        person_id = None
        if person_uuid:
            cur.execute(
                "SELECT person_id FROM known_persons WHERE person_uuid = %s",
                (person_uuid,)
            )
            row = cur.fetchone()
            if row:
                person_id = row["person_id"]
 
        cur.execute("""
            INSERT INTO interaction_timeline
                (patient_id, person_id, recognition_status,
                 confidence_score, snapshot_path, device_info)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (patient_id, person_id, recognition_status,
              confidence_score, snapshot_path, device_info))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] log_interaction(): {e}")
        return None
    finally:
        cur.close(); conn.close()
 
 
def get_timeline(patient_id: int, limit: int = 50) -> list:
    """
    Returns recent interaction history for a patient, newest first.
    Used by the frontend Timeline panel.
    """
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT it.interaction_id,
                   it.recognition_status,
                   it.confidence_score,
                   it.interaction_at,
                   it.snapshot_path,
                   it.device_info,
                   kp.full_name    AS person_name,
                   kp.relationship,
                   kp.person_uuid  AS person_id
            FROM   interaction_timeline it
            LEFT   JOIN known_persons kp ON it.person_id = kp.person_id
            WHERE  it.patient_id = %s
            ORDER  BY it.interaction_at DESC
            LIMIT  %s
        """, (patient_id, limit))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# UNKNOWN FACE QUEUE
# ─────────────────────────────────────────────
 
def queue_unknown_face(patient_id: int,
                       embedding: np.ndarray = None,
                       snapshot_path: str = None) -> Optional[int]:
    """Queues an unrecognized face for caregiver review."""
    conn = _connect()
    cur  = conn.cursor()
    try:
        emb_json = None
        if embedding is not None:
            emb_list = (
                embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else list(embedding)
            )
            emb_json = json.dumps(emb_list)
 
        cur.execute("""
            INSERT INTO unknown_face_queue (patient_id, snapshot_path, embedding_data)
            VALUES (%s, %s, %s)
        """, (patient_id, snapshot_path, emb_json))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] queue_unknown_face(): {e}")
        return None
    finally:
        cur.close(); conn.close()
 
 
def get_pending_queue(patient_id: int) -> list:
    """Returns pending unknown faces awaiting caregiver review."""
    conn = _connect()
    cur  = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT queue_id, snapshot_path, created_at
            FROM   unknown_face_queue
            WHERE  patient_id = %s AND status = 'pending'
            ORDER  BY created_at DESC
        """, (patient_id,))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()
 
 
def resolve_queue_item(queue_id: int, status: str,
                       reviewed_by: int) -> bool:
    """Caregiver accepts or rejects a queued face. status: 'accepted' | 'rejected'"""
    conn = _connect()
    cur  = conn.cursor()
    try:
        cur.execute("""
            UPDATE unknown_face_queue
               SET status = %s, reviewed_by = %s, reviewed_at = %s
             WHERE queue_id = %s
        """, (status, reviewed_by, datetime.now(), queue_id))
        conn.commit()
        return cur.rowcount > 0
    except Error as e:
        conn.rollback()
        print(f"[DB ERROR] resolve_queue_item(): {e}")
        return False
    finally:
        cur.close(); conn.close()
 
 
# ─────────────────────────────────────────────
# QUICK-START DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import bcrypt
 
    print("\n" + "=" * 55)
    print("  MITRA DB — Quick-Start Demo")
    print("=" * 55)
 
    # 1. Setup
    print("\n[1] Setting up database & tables...")
    setup(sql_file="mitra_schema.sql")   # falls back to inline schema if file absent
 
    # 2. Caregiver
    print("\n[2] Creating caregiver...")
    hashed = bcrypt.hashpw(b"SecurePass123", bcrypt.gensalt()).decode()
    cg_id  = add_caregiver("Priya Mehta", "priya@mitra.ai", hashed, "+91-9000000001")
    print(f"    Caregiver ID: {cg_id}")
 
    # 3. Patient
    print("\n[3] Adding patient...")
    pt_id = add_patient(cg_id, "Suresh Mehta", "1948-06-20",
                         "Mild-to-moderate dementia. Responds to music.")
    print(f"    Patient ID: {pt_id}")
 
    # 4. Register a known person (simulates pipeline.py registering via face scan)
    print("\n[4] Registering known person with embedding...")
    dummy_emb = np.array([round(i * 0.007, 4) for i in range(512)])  # ArcFace = 512-D
    p_uuid = register_person(
        patient_id   = pt_id,
        name         = "Priya Mehta",
        relationship = "Daughter",
        notes        = "Visits every Sunday. Likes cricket conversations.",
        embedding    = dummy_emb,
        caregiver_id = cg_id
    )
    print(f"    Person UUID: {p_uuid}")
 
    # 5. Simulate load_db (as matcher.py calls it on startup)
    print("\n[5] Loading DB for face-matching engine...")
    db = load_db(patient_id=pt_id)
    for name, d in db.items():
        print(f"    ✓ {name} | relationship: {d['relationship']} | emb dim: {len(d['embedding'])}")
 
    # 6. Log interactions
    print("\n[6] Logging recognition events to timeline...")
    log_interaction(pt_id, "recognized", person_uuid=p_uuid,
                    confidence_score=0.93, device_info="Pi-Cam-01")
    log_interaction(pt_id, "unknown", confidence_score=0.28)
 
    # 7. Timeline
    print("\n[7] Recent timeline:")
    for entry in get_timeline(pt_id, limit=5):
        ts   = entry["interaction_at"]
        name = entry["person_name"] or "Unknown"
        conf = entry["confidence_score"]
        print(f"    [{ts}] {entry['recognition_status'].upper():12s} — {name} (conf: {conf})")
 
    # 8. Queue unknown face
    print("\n[8] Queuing unknown face for caregiver review...")
    q_id = queue_unknown_face(pt_id, embedding=dummy_emb[:64],
                               snapshot_path="/snapshots/unknown_001.jpg")
    pending = get_pending_queue(pt_id)
    print(f"    Pending items: {len(pending)}")
 
    print("\n" + "=" * 55)
    print("  Done! MySQL database is live.")
    print("=" * 55 + "\n")
 
