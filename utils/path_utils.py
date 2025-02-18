from pathlib import Path

def get_project_root() -> Path:
    """Find the project root directory by looking for .git or ClusterSense directory
    
    Returns:
        Path: Project root directory path
        
    Raises:
        ValueError: If project root cannot be found
    """
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:  # Stop at root directory
        if (current_path / '.git').exists():
            return current_path
        if current_path.name == 'ClusterSense':  # Also check for repo name
            return current_path
        current_path = current_path.parent
    raise ValueError("Could not find ClusterSense project root")

def get_results_dir() -> Path:
    """Get the results directory path, creating it if it doesn't exist
    
    Returns:
        Path: Results directory path
    """
    results_dir = get_project_root() / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
