import os
import time
import glob
import argparse
import json
from huggingface_hub import HfApi

REPO_ID = "auto-cap/moe-cap-results"

POLL_INTERVAL = 10
STATE_FILE = ".moe_cap_watcher_state.json"


def get_api_client():
    token = os.environ.get("HF_TOKEN")
    if token:
        print("[Watcher] Detected HF_TOKEN environment variable, authenticating...")
        return HfApi(token=token)
    else:
        print("[Watcher] HF_TOKEN environment variable not found.")
        print("   Attempting to initialize without token (may fail for uploads)...")
        return HfApi()

def load_state(state_file):
    """Load uploaded files state from disk"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                uploaded = set(data.get('uploaded_files', []))
                print(f"[State] Loaded {len(uploaded)} previously uploaded files from state file")
                return uploaded
        except Exception as e:
            print(f"[State] Warning: Could not load state file: {e}")
            return set()
    return set()

def save_state(state_file, uploaded_files):
    """Save uploaded files state to disk"""
    try:
        with open(state_file, 'w') as f:
            json.dump({'uploaded_files': list(uploaded_files)}, f, indent=2)
    except Exception as e:
        print(f"[State] Warning: Could not save state file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor directory and upload files to HuggingFace")
    parser.add_argument(
        "--monitor-dir",
        type=str,
        required=True,
        help="Local output directory to monitor for new files"
    )
    args = parser.parse_args()
    
    LOCAL_OUTPUT_DIR = os.path.abspath(args.monitor_dir)
    
    # Use XDG state directory or fallback to home directory
    state_dir = os.path.join(os.path.expanduser("~"), ".local", "state", "watcher")
    os.makedirs(state_dir, exist_ok=True)
    
    # Create unique state file based on monitored directory path
    dir_hash = str(abs(hash(LOCAL_OUTPUT_DIR)))
    state_file = os.path.join(state_dir, f"{STATE_FILE}.{dir_hash}")
    
    api = get_api_client()
    uploaded_files = load_state(state_file)

    print(f"Starting directory monitor: {LOCAL_OUTPUT_DIR}")
    print(f"Target repository: {REPO_ID}")


    if not os.path.exists(LOCAL_OUTPUT_DIR):
        print(f"Directory {LOCAL_OUTPUT_DIR} does not exist yet. Waiting...")
        while not os.path.exists(LOCAL_OUTPUT_DIR):
            time.sleep(5)
        print(f"Directory found: {LOCAL_OUTPUT_DIR}")

    print(f"State file: {state_file}")
    
    try:
        print("Scanning directory...", end="")
        initial_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "**/*.json"), recursive=True)
        existing_count = len([f for f in initial_files if os.path.abspath(f) in uploaded_files])
        new_count = len([f for f in initial_files if os.path.abspath(f) not in uploaded_files])
        print(f" Found {len(initial_files)} files ({existing_count} already uploaded, {new_count} new)")
    except Exception as e:
        print(f" (Scan warning: {e})")

    while True:
        try:
            current_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "**/*.json"), recursive=True)
            
            for file_path in current_files:
                abs_path = os.path.abspath(file_path)
                
                if abs_path not in uploaded_files:
                    relative_path = os.path.relpath(abs_path, LOCAL_OUTPUT_DIR)
                    
                    print(f"New file found: {relative_path}")
                    print(f"   Uploading...", end="", flush=True)
                    
                    try:
                        api.upload_file(
                            path_or_fileobj=abs_path,
                            path_in_repo=relative_path, 
                            repo_id=REPO_ID,
                            repo_type="dataset"
                        )
                        uploaded_files.add(abs_path)
                        save_state(state_file, uploaded_files)
                        print(" Success")
                    except Exception as upload_err:
                        print(f" Failed: {upload_err}")
        
        except Exception as e:
            print(f"Polling error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()