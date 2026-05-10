import ast
import os


def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return None

    output = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            output.append(f"  Class: {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [arg.arg for arg in item.args.args]
                    output.append(f"    Method: {item.name}({', '.join(args)})")
        elif isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            output.append(f"  Function: {node.name}({', '.join(args)})")

    return "\n".join(output) if output else None


def generate_map(root_dir):
    exclude = {'.git', 'venv', '.idea', '__pycache__', 'node_modules'}

    print("# Структура проекта и API\n")

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude]

        for file in files:
            if file.endswith('.py') and file != 'map_project.py':
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, root_dir)

                content = analyze_file(filepath)
                if content:
                    print(f"### File: {rel_path}")
                    print(content)
                    print("\n")


if __name__ == "__main__":
    generate_map('.')
