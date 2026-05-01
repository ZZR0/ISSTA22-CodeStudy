import os
import json
from pathlib import Path

def setup():
    # 1. Create directories
    base_dir = Path("./data")
    systems_dir = base_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    # 2. Write the Python source file with continuous, functional code
    py_code = """class SecurityHelper:
    def __init__(self):
        self.audit_log = []

    def process_realm_access(self, access_payload: dict, user_id: str) -> list:
        # PRE-REGION
        roles = []
        timestamp = "2026-03-01T21:07:25Z"
        is_valid = access_payload.get("active", False)
        
        # WITHIN-REGION (Clone 1)
        if isinstance(access_payload, dict) and is_valid:
            extracted = access_payload.get("roles", [])
            if isinstance(extracted, list):
                for r in extracted:
                    roles.append(r.upper())
        
        # POST-REGION
        self.audit_log.append(f"Realm access processed for {user_id} at {timestamp}")
        return roles

    def process_client_access(self, access_payload: dict, client_id: str) -> list:
        # PRE-REGION
        roles = []
        timestamp = "2026-03-01T21:07:25Z"
        is_valid = access_payload.get("enabled", False)
        
        # WITHIN-REGION (Clone 2)
        if isinstance(access_payload, dict) and is_valid:
            extracted = access_payload.get("roles", [])
            if isinstance(extracted, list):
                for r in extracted:
                    roles.append(r.upper())
        
        # POST-REGION
        self.audit_log.append(f"Client access processed for {client_id} at {timestamp}")
        return roles
"""
    py_path = systems_dir / "security_helper.py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_code)
    print(f"Created: {py_path}")

    # 3. Write the JSONL file mapping to the new line numbers
    # Clone 1 is lines 12-16
    # Clone 2 is lines 29-33
    jsonl_path = base_dir / "sample_python.jsonl"
    clone_code_snippet = """        if isinstance(access_payload, dict) and is_valid:
            extracted = access_payload.get("roles", [])
            if isinstance(extracted, list):
                for r in extracted:
                    roles.append(r.upper())"""

    clone_data = {
        "classid": 110, 
        "nclones": 2, 
        "similarity": 100.0, 
        "sources": [
            {
                "file": "systems/security_helper.py", 
                "range": "12-16", 
                "nlines": 5, 
                "pcid": "4512", 
                "code": clone_code_snippet, 
                "func_id": "myapp_110_45"
            }, 
            {
                "file": "systems/security_helper.py", 
                "range": "29-33", 
                "nlines": 5, 
                "pcid": "4513", 
                "code": clone_code_snippet, 
                "func_id": "myapp_110_46"
            }
        ]
    }
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(clone_data) + "\n")
    print(f"Created: {jsonl_path}")

if __name__ == "__main__":
    setup()