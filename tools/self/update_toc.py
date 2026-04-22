#!/usr/bin/env python3
"""
Script to generate a Tree Diagram Table of Contents for all files
and update README.md with the tree structure using HTML tags.
"""

import os
from pathlib import Path
from typing import List, Set, Dict, Optional
import argparse
import re

class TreeTOCGenerator:
    """Generate tree diagram TOC for README files using HTML."""
    
    def __init__(self, root_dir: str = ".", exclude_dirs: List[str] = None, 
                 exclude_files: List[str] = None, max_depth: Optional[int] = None):
        """
        Initialize the Tree TOC Generator.
        
        Args:
            root_dir: Root directory to scan
            exclude_dirs: List of directory names to exclude
            exclude_files: List of file names to exclude
            max_depth: Maximum depth to traverse (None for unlimited)
        """
        self.root_dir = Path(root_dir).resolve()
        self.exclude_dirs = set(exclude_dirs or [
            '.git', '__pycache__', 'venv', 'env', 'node_modules', 
            '.venv', '.idea', '.vscode', 'dist', 'build', 'temp',
            '.ipynb_checkpoints'  # Added this common exclusion
        ])
        self.exclude_files = set(exclude_files or [
            'README.md', '.gitignore', 'LICENSE', '.DS_Store', 
            'Thumbs.db', 'update_toc.py', 'tree_toc.py'
        ])
        self.max_depth = max_depth
        
    def should_exclude_dir(self, dir_path: Path) -> bool:
        """Check if directory should be excluded."""
        # Check if any excluded directory name is in the path
        for excluded in self.exclude_dirs:
            if excluded in dir_path.parts:
                return True
        return False
    
    def should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded."""
        return file_path.name in self.exclude_files
    
    def get_relative_path(self, path: Path) -> str:
        """Get path relative to root directory."""
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)
    
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#39;'))
    
    def build_tree(self, current_dir: Path = None, depth: int = 0, 
                   prefix: str = "", is_last: bool = True, use_links: bool = True) -> List[str]:
        """
        Recursively build the directory tree.
        
        Args:
            current_dir: Current directory to process
            depth: Current depth level
            prefix: Prefix for tree formatting
            is_last: Whether this is the last item at current level
            use_links: Whether to create clickable links
            
        Returns:
            List of tree lines
        """
        if current_dir is None:
            current_dir = self.root_dir
        
        if self.max_depth is not None and depth > self.max_depth:
            return []
        
        lines = []
        
        # Get all items in current directory
        try:
            items = sorted(current_dir.iterdir())
        except PermissionError:
            return lines
        
        # Separate directories and files
        dirs = [item for item in items if item.is_dir() and not self.should_exclude_dir(item)]
        files = [item for item in items if item.is_file() and not self.should_exclude_file(item)]
        
        # Process all items
        all_items = dirs + files
        for idx, item in enumerate(all_items):
            is_last_item = (idx == len(all_items) - 1)
            
            # Choose the connector
            connector = "└── " if is_last_item else "├── "
            
            # Create the line
            if item.is_dir():
                # Directory
                line = f"{prefix}{connector}📁 {item.name}/"
                lines.append(line)
                
                # Recursively process subdirectory
                extension = "    " if is_last_item else "│   "
                sub_lines = self.build_tree(
                    item, 
                    depth + 1, 
                    prefix + extension,
                    is_last_item,
                    use_links
                )
                lines.extend(sub_lines)
            else:
                # File
                if use_links:
                    relative_path = self.get_relative_path(item)
                    # URL encode spaces and special characters
                    link_path = relative_path.replace(' ', '%20')
                    line = f'{prefix}{connector}📄 <a href="{link_path}">{item.name}</a>'
                else:
                    line = f'{prefix}{connector}📄 {item.name}'
                lines.append(line)
        
        return lines
    
    def generate_tree_toc(self, title: str = "🌳 Directory Tree", use_links: bool = True) -> str:
        """
        Generate the complete tree diagram TOC.
        
        Args:
            title: Title for the TOC section
            use_links: Whether to include clickable links
            
        Returns:
            HTML formatted tree TOC
        """
        # Use a styled pre block that preserves formatting
        html_parts = [
            f"<h2>{title}</h2>",
            '<div class="directory-tree" style="font-family: monospace; white-space: pre; overflow-x: auto;">',
            f'<strong>{self.root_dir.name}/</strong><br/>'
        ]
        
        # Generate tree structure
        tree_lines = self.build_tree(use_links=use_links)
        
        # Join lines with <br/> tags and proper indentation preservation
        for line in tree_lines:
            # Replace spaces with &nbsp; to preserve them in HTML
            # But keep the tree characters as-is
            formatted_line = line.replace(' ', '&nbsp;')
            formatted_line = formatted_line.replace("<a&nbsp;", "<a ")
            html_parts.append(f"{formatted_line}<br/>")
        
        html_parts.append('</div>')
        
        if use_links:
            html_parts.append('<p><em>Click on any 📄 file to view it</em></p>')
        
        return "\n".join(html_parts)

def update_readme_with_tree(readme_path: str = "README.md", 
                           tree_generator: Optional[TreeTOCGenerator] = None,
                           use_links: bool = True,
                           toc_marker_start: str = "<!-- TOC_START -->",
                           toc_marker_end: str = "<!-- TOC_END -->") -> bool:
    """
    Update README.md with HTML tree diagram TOC.
    
    Args:
        readme_path: Path to README.md
        tree_generator: TreeTOCGenerator instance
        use_links: Whether to include clickable links
        toc_marker_start: Start marker for TOC
        toc_marker_end: End marker for TOC
        
    Returns:
        True if successful
    """
    # Create README if it doesn't exist
    if not os.path.exists(readme_path):
        print(f"⚠️  {readme_path} not found. Creating new file...")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"<h1>Project Documentation</h1>\n\n{toc_marker_start}\n{toc_marker_end}\n\n<h2>About</h2>\n<p>Add your project description here.</p>\n")
    
    # Create default tree generator if none provided
    if tree_generator is None:
        tree_generator = TreeTOCGenerator()
    
    # Generate tree TOC
    tree_toc = tree_generator.generate_tree_toc(use_links=use_links)
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for markers
    if toc_marker_start not in content or toc_marker_end not in content:
        print(f"⚠️  TOC markers not found in {readme_path}")
        print(f"Please add these markers to your README:")
        print(f"  {toc_marker_start}")
        print(f"  {toc_marker_end}")
        return False
    
    # Replace content between markers
    pattern = re.compile(
        f"{re.escape(toc_marker_start)}.*?{re.escape(toc_marker_end)}",
        re.DOTALL
    )
    
    new_content = pattern.sub(
        f"{toc_marker_start}\n\n{tree_toc}\n\n{toc_marker_end}",
        content
    )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Successfully updated {readme_path}")
    return True

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate Tree Diagram Table of Contents for README using HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generate tree with links
  python tree_toc.py
  
  # Limit depth to 3 levels
  python tree_toc.py --max-depth 3
  
  # Simple tree without clickable links
  python tree_toc.py --simple
  
  # Exclude additional directories
  python tree_toc.py --exclude-dirs tests docs temp
  
  # Custom README location
  python tree_toc.py --readme docs/README.md
        """
    )
    
    parser.add_argument(
        "-r", "--readme",
        default="README.md",
        help="Path to README file (default: README.md)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory depth to display"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Generate simple tree without clickable links"
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        help="Additional directories to exclude"
    )
    parser.add_argument(
        "--exclude-files",
        nargs="+",
        help="Additional files to exclude"
    )
    
    args = parser.parse_args()
    
    # Build exclude lists
    exclude_dirs = ['.git', '__pycache__', 'venv', 'env', 'node_modules', 
                    '.venv', '.idea', '.vscode', 'dist', 'build', 'temp']
    if args.exclude_dirs:
        exclude_dirs.extend(args.exclude_dirs)
    
    exclude_files = ['README.md', '.gitignore', 'LICENSE', '.DS_Store', 'Thumbs.db']
    if args.exclude_files:
        exclude_files.extend(args.exclude_files)
    
    # Create tree generator
    generator = TreeTOCGenerator(
        root_dir=".",
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
        max_depth=args.max_depth
    )
    
    # Update README
    success = update_readme_with_tree(
        readme_path=args.readme,
        tree_generator=generator,
        use_links=not args.simple
    )
    
    if not success:
        exit(1)
    
    # Print summary
    print("\n📊 Summary:")
    print(f"  • Root directory: {generator.root_dir}")
    if args.max_depth:
        print(f"  • Max depth: {args.max_depth}")
    print(f"  • Link type: {'Clickable' if not args.simple else 'Plain text'}")
    print(f"  • Excluded dirs: {', '.join(generator.exclude_dirs)}")
    print(f"  • Format: HTML with preserved whitespace")

if __name__ == "__main__":
    main()