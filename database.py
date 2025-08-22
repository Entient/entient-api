import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL')

def init_db():
    """Create tables if they don't exist"""
    if not DATABASE_URL:
        print("Warning: No DATABASE_URL, using in-memory storage")
        return
    
    # Railway's DATABASE_URL starts with 'postgres://' but psycopg2 needs 'postgresql://'
    db_url = DATABASE_URL.replace('postgres://', 'postgresql://')
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS seals (
            seal_id VARCHAR(255) PRIMARY KEY,
            content_hash TEXT,
            classification VARCHAR(50),
            fitness_score FLOAT,
            strategy_used VARCHAR(100),
            timestamp_utc TIMESTAMP,
            discovery_content TEXT,
            hmac_signature TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()

def save_seal(seal, discovery_content):
    """Save seal to database"""
    if not DATABASE_URL:
        return False
    
    db_url = DATABASE_URL.replace('postgres://', 'postgresql://')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    try:
        cur.execute('''
            INSERT INTO seals (seal_id, content_hash, classification, fitness_score, 
                             strategy_used, timestamp_utc, discovery_content, hmac_signature)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            seal.session_uuid,
            seal.content_hash,
            seal.discovery_classification.value,
            seal.fitness_score,
            seal.strategy_used,
            seal.timestamp_utc,
            discovery_content,
            seal.hmac_signature
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def get_seal(seal_id):
    """Retrieve seal from database"""
    if not DATABASE_URL:
        return None
    
    db_url = DATABASE_URL.replace('postgres://', 'postgresql://')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute('SELECT * FROM seals WHERE seal_id = %s', (seal_id,))
    result = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return result

def get_all_seals():
    """Get all seals from database"""
    if not DATABASE_URL:
        return []
    
    db_url = DATABASE_URL.replace('postgres://', 'postgresql://')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute('SELECT * FROM seals ORDER BY created_at DESC LIMIT 100')
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results