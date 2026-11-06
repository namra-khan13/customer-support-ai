import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "supportiq.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                customer TEXT,
                message TEXT NOT NULL,
                category TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                priority TEXT NOT NULL,
                category_confidence REAL NOT NULL,
                suggested_response TEXT NOT NULL
            )
        """)

def save_ticket(customer, message, result):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO tickets
            (customer, message, category, sentiment, priority, category_confidence, suggested_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            customer or "Anonymous", message, result["category"], result["sentiment"],
            result["priority"], result["category_confidence"], result["suggested_response"]
        ))

def load_tickets():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM tickets ORDER BY created_at DESC", conn)
