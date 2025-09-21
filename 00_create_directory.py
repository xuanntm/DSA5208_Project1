import os

def create_project_folders(base="data"):
    structure = [
        os.path.join(base, "logs"),
        os.path.join(base, "input"),
        os.path.join(base, "output", "cleanup_data"),
        os.path.join(base, "output", "model"),
        os.path.join(base, "output", "split_data"),
        os.path.join(base, "output", "training"),
    ]

    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

if __name__ == "__main__":
    create_project_folders()
