import os
import sys
import difflib
import ast

def compare_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    with open(output_file, 'w') as out:
        # Text diff
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=file1, tofile=file2, lineterm=''))
        if diff:
            out.write("Text differences:\n")
            for line in diff:
                out.write(line + '\n')
        else:
            out.write("No text differences.\n")
        
        # AST comparison for major differences
        try:
            tree1 = ast.parse(''.join(lines1), file1)
            tree2 = ast.parse(''.join(lines2), file2)
            
            functions1 = [node.name for node in ast.walk(tree1) if isinstance(node, ast.FunctionDef)]
            functions2 = [node.name for node in ast.walk(tree2) if isinstance(node, ast.FunctionDef)]
            
            added_functions = set(functions2) - set(functions1)
            removed_functions = set(functions1) - set(functions2)
            
            if added_functions:
                out.write(f"Added functions: {added_functions}\n")
            if removed_functions:
                out.write(f"Removed functions: {removed_functions}\n")
            
            classes1 = [node.name for node in ast.walk(tree1) if isinstance(node, ast.ClassDef)]
            classes2 = [node.name for node in ast.walk(tree2) if isinstance(node, ast.ClassDef)]
            
            added_classes = set(classes2) - set(classes1)
            removed_classes = set(classes1) - set(classes2)
            
            if added_classes:
                out.write(f"Added classes: {added_classes}\n")
            if removed_classes:
                out.write(f"Removed classes: {removed_classes}\n")
            
        except SyntaxError as e:
            out.write(f"Syntax error in parsing: {e}\n")

if __name__ == "__main__":
    file1 = "C:\\Users\\dunc3\\OneDrive\\Documents\\School\\Current Semester\\Physics Project\\main\\scripts\\data_creation\\Four_Layer_IB_Eg_1.67.py"
    file2 = "C:\\Users\dunc3\OneDrive\Documents\School\Current Semester\Physics Project\main\Fatemeh\data\fourlayer_example.py"
    output_file = "comparison_results.txt"
    
    if not os.path.exists(file1):
        print(f"File {file1} does not exist.")
        exit(1)
    if not os.path.exists(file2):
        print(f"File {file2} does not exist.")
        exit(1)
    
    compare_files(file1, file2, output_file)
    print(f"Comparison results saved to {output_file}")
