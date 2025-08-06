import subprocess
import os

def _download_spacy():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("en_core_web_sm model already installed")
    except (ImportError, OSError):
        print("Installing en_core_web_sm model...")
        subprocess.run("python -m spacy download en_core_web_sm", shell=True)

def _download_nltk():
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')

def maybe_download_packages(outputs_dir: str):
    if not os.path.exists(outputs_dir):
        _download_spacy()
        _download_nltk()
