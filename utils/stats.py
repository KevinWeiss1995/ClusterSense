import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

def get_git_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    raise RuntimeError("Not in a git repository")

def load_data():
    git_root = get_git_root()
    df = pd.read_csv(git_root / "data" / "HPC_nodes.csv")
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    return df

def analyze_failure_intervals(df):
    """Calculate time between failures for each node"""
    failures = df[df['State'].str.contains('unavailable', na=False)]
    intervals = defaultdict(list)
    
    for node in failures['Node'].unique():
        node_failures = failures[failures['Node'] == node].sort_values('Time')
        if len(node_failures) > 1:
            time_diffs = node_failures['Time'].diff().dropna()
            intervals[node] = time_diffs.dt.total_seconds() / 3600  # Convert to hours
    
    print("\nFailure Intervals Analysis:")
    for node, times in intervals.items():
        if len(times) > 0:
            print(f"{node}: Avg {np.mean(times):.2f} hours between failures")
    
    return intervals

def analyze_related_failures(df):
    """Group failures that occur within 5 minutes of each other"""
    failures = df[df['State'].str.contains('unavailable', na=False)].sort_values('Time')
    failure_clusters = []
    current_cluster = []
    
    for idx, row in failures.iterrows():
        if not current_cluster:
            current_cluster = [row]
        else:
            time_diff = (row['Time'] - current_cluster[-1]['Time']).total_seconds()
            if time_diff <= 300:  # 5 minutes
                current_cluster.append(row)
            else:
                if len(current_cluster) > 1:
                    failure_clusters.append(current_cluster)
                current_cluster = [row]
    
    print("\nRelated Failures Analysis:")
    print(f"Found {len(failure_clusters)} clusters of related failures")
    for i, cluster in enumerate(failure_clusters[:5]):  # Show first 5 clusters
        print(f"\nCluster {i+1}: {len(cluster)} nodes failed within 5 minutes")
        for failure in cluster:
            print(f"  {failure['Node']} at {failure['Time']}")
    
    return failure_clusters

def analyze_network_changes(df):
    """Analyze network state changes before hardware failures"""
    network_changes = df[df['State'].str.contains('niff', na=False)]
    failures = df[df['State'].str.contains('unavailable', na=False)]
    
    precursor_events = defaultdict(int)
    window = pd.Timedelta(hours=1)
    
    for idx, failure in failures.iterrows():
        prior_changes = network_changes[
            (network_changes['Time'] > failure['Time'] - window) &
            (network_changes['Time'] < failure['Time']) &
            (network_changes['Node'] == failure['Node'])
        ]
        precursor_events[failure['Node']] += len(prior_changes)
    
    print("\nNetwork Changes Analysis:")
    print("Network changes in hour before failure:")
    for node, count in precursor_events.items():
        if count > 0:
            print(f"{node}: {count} changes")
    
    return precursor_events

def analyze_hardware_components(df):
    """Map hardware component relationships and failures"""
    components = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        if 'Component' in row['Content']:
            # Extract component name from content
            comp = row['Content'].split('Component')[1].split()[1].strip('"\\')
            components[row['Node']][comp] += 1
    
    print("\nHardware Component Analysis:")
    print("Most problematic components by node:")
    for node, comps in components.items():
        if comps:
            worst_comp = max(comps.items(), key=lambda x: x[1])
            print(f"{node}: {worst_comp[0]} failed {worst_comp[1]} times")
    
    return components

def main():
    df = load_data()
    print("Analyzing HPC node failure data...")
    
    intervals = analyze_failure_intervals(df)
    clusters = analyze_related_failures(df)
    precursors = analyze_network_changes(df)
    components = analyze_hardware_components(df)
    
    # Optional: Create visualizations
    plt.figure(figsize=(12, 6))
    plt.hist([x for times in intervals.values() for x in times], bins=50)
    plt.title('Distribution of Time Between Failures')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    
    git_root = get_git_root()
    plt.savefig(git_root / "data" / "failure_intervals.png")
    plt.close()

if __name__ == "__main__":
    main() 