import os

def get_file_path(filename: str) -> str:
    """
    Retourne le chemin absolu pour accéder au fichier.
    En local, on prend ./file/filename
    Sur Azure, on prend /file/filename
    """
    if os.path.exists("/file"):   # Si le dossier monté existe (Azure)
        base_dir = "/file"
    else:                         # Sinon en local
        base_dir = os.path.join(os.path.dirname(__file__), "file")
    return os.path.join(base_dir, filename)