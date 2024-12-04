from pathlib import Path
import sys
import os

def generate_tree(
    directory: Path,
    ignore_patterns: list = ['__pycache__', '.git', '.pytest_cache', '*.pyc', '.DS_Store', '*.idea'],
    prefix: str = '',
    is_last: bool = True
) -> str:
    directory = Path(directory)
    output = []
    
    if prefix:
        output.append(f"{prefix}{'└── ' if is_last else '├── '}{directory.name}")
    else:
        output.append(str(directory))
    
    items = []
    for item in directory.iterdir():
        if any(item.match(pattern) for pattern in ignore_patterns):
            continue
        items.append(item)
    
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    
    for i, item in enumerate(items):
        if prefix:
            child_prefix = prefix + ('    ' if is_last else '│   ')
        else:
            child_prefix = prefix
        
        if item.is_dir():
            output.extend(
                generate_tree(
                    item,
                    ignore_patterns,
                    child_prefix,
                    i == len(items) - 1
                )
            )
        else:
            output.append(f"{child_prefix}{'└── ' if i == len(items) - 1 else '├── '}{item.name}")
    
    return output

def print_and_save_tree(root_path: str, output_file: str = None):
    root_path = Path(root_path).resolve()
    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist!")
        return
    
    tree_lines = generate_tree(root_path)
    
    # Print to console
    print("\nProject Structure:")
    print("-----------------")
    for line in tree_lines:
        print(line)
    
    # Save to file if specified
    if output_file:
        output_path = root_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Project Structure:\n")
            f.write("-----------------\n")
            f.write('\n'.join(tree_lines))
        print(f"\nStructure saved to {output_path}")

if __name__ == "__main__":
    # Get the project root directory (Car_Tracking_Algorithms)
    root_dir = Path(__file__).resolve().parent.parent
    output_file = "project_structure.txt"
    
    print_and_save_tree(root_dir, output_file)