#!/usr/bin/env python3

import re
import sys

def fix_match_statements(content):
    """Fix missing match self { statements in generated Rust code"""
    
    # Pattern to find functions missing match self {
    # Looks for: fn name(&self) -> Type { followed by ModelName::Variant
    pattern = r'(fn [^{]+\{)\s*\n(\s+)([A-Z][a-zA-Z]*Model::[A-Za-z0-9_]+)'
    
    def replace_func(match):
        func_signature = match.group(1)  # fn name(&self) -> Type {
        indent = match.group(2)          # whitespace before the enum
        enum_line = match.group(3)       # ModelName::Variant
        
        # Add match self { and adjust indentation
        return f"{func_signature}\n{indent}match self {{\n{indent}    {enum_line}"
    
    # Apply the replacement
    fixed_content = re.sub(pattern, replace_func, content, flags=re.MULTILINE)
    
    # Now we need to fix the closing braces - find functions that now need an extra }
    # Look for match blocks that don't have proper closing
    lines = fixed_content.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        result_lines.append(line)
        
        # If this line contains "match self {" and we're in a function
        if "match self {" in line and "fn " in line:
            # Find the end of this function by looking for the closing brace
            # at the same indentation level as the fn
            func_indent = len(line) - len(line.lstrip())
            
            # Look ahead to find where this function ends
            j = i + 1
            brace_count = 1  # We opened with match self {
            
            while j < len(lines) and brace_count > 0:
                next_line = lines[j]
                result_lines.append(next_line)
                
                # Count braces to track nesting
                brace_count += next_line.count('{') - next_line.count('}')
                j += 1
            
            # If we still have unmatched braces, we need to add a closing brace
            if brace_count > 0:
                # Find the right indentation for the closing brace
                last_line = result_lines[-1] if result_lines else ""
                if last_line.strip() == "}":
                    # Add another closing brace for the match
                    match_indent = line[:line.find("match")]
                    result_lines.append(f"{match_indent}}}")
            
            i = j - 1
        
        i += 1
    
    return '\n'.join(result_lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_match_statements.py <file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        fixed_content = fix_match_statements(content)
        
        with open(filename, 'w') as f:
            f.write(fixed_content)
        
        print(f"Fixed match statements in {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)