import subprocess


def install_spacy():
    subprocess.run("python -m spacy download en_core_web_sm", shell=True)

def install_nltk():
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
