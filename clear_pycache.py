import os
import shutil

def clear_pycache(start_path):
    """
    Recursively deletes all __pycache__ directories found within the start_path.
    
    Args:
        start_path (str): The root directory to start searching from.
    """
    if not os.path.exists(start_path):
        print(f"Directory not found: {start_path}")
        return

    print(f"Scanning for __pycache__ directories in: {start_path}")
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(start_path):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                # Remove the directory and all its contents
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                
                # Remove from dirs list to prevent walking into it (since it's deleted)
                dirs.remove("__pycache__")
            except Exception as e:
                print(f"Failed to remove {pycache_path}: {e}")

if __name__ == "__main__":
    # Define the target directory (src folder relative to this script)
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    
    clear_pycache(target_dir)
    print("Done.")