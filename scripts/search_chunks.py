import os
import sys
import sqlite3

# Ensure stdout uses UTF-8 encoding on Windows to prevent character encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def search_chunks(keyword, limit=5):
    # Find path to rag.db database in project root
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag.db"))
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Search for chunks containing keyword in 'text'
    query = "SELECT id, source_file_name, text FROM chunks WHERE text LIKE ? LIMIT ?"
    cursor.execute(query, (f"%{keyword}%", limit))
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No chunks found containing keyword: '{keyword}'")
        conn.close()
        return
        
    print(f"=== FOUND {len(rows)} CHUNKS CONTAINING KEYWORD '{keyword}' ===\n")
    for idx, row in enumerate(rows, 1):
        chunk_id, source_file, text = row
        print(f"{idx}. SOURCE FILE: {source_file}")
        print(f"   CHUNK ID   : {chunk_id}")
        
        # Find keyword index to display snippet
        keyword_idx = text.lower().find(keyword.lower())
        if keyword_idx != -1:
            start = max(0, keyword_idx - 60)
            end = min(len(text), keyword_idx + len(keyword) + 140)
            snippet = text[start:end].strip().replace('\n', ' ')
            print(f"   CONTENT    : ... {snippet} ...")
        else:
            print(f"   CONTENT    : {text[:200].strip().replace('\n', ' ')}...")
            
        print("-" * 80)
        
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_chunks.py <keyword>")
        sys.exit(1)
        
    keyword = " ".join(sys.argv[1:])
    search_chunks(keyword)
