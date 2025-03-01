import csv
from pathlib import Path
import os

def get_git_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    raise RuntimeError("Not in a git repository")

def convert_to_csv(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['LogId', 'Node', 'Component', 'State', 'Time', 'Flag', 'Content'])
        
        for line in infile:
            # Skip empty lines or lines starting with ...
            if not line.strip() or line.strip().startswith('...'):
                continue
                
            # Split on first 6 spaces to keep content intact
            parts = line.strip().split(None, 6)
            if len(parts) != 7:
                continue
            
            # Only keep entries where Node starts with "node-"
            if parts[1].startswith('node-'):
                writer.writerow(parts)

if __name__ == "__main__":
    git_root = get_git_root()
    input_path = git_root / "data" / "HPC.log"
    output_path = git_root / "data" / "HPC_nodes.csv"
    convert_to_csv(input_path, output_path)