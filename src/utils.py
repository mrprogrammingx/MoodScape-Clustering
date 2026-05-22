import os
def create_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
def print_separator():
    print("-" * 50)
