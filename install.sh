#!/bin/bash

# The PyPI version submodlib seems to be broken, so
# one has to resort to manual installation.
mkdir external_metrics
cd external_metrics
echo "Installing submodlib"
git clone https://github.com/decile-team/submodlib.git
cd submodlib
# Cannot run `pip install -r requirements.txt` due to versions mismatch
pip install sphinxcontrib-bibtex pybind11>=2.6.0 scikit-learn scipy
pip install . --no-deps
cd ..
rm -rf submodlib

# Install external metrics
echo "Installing AlignScore..."
git clone https://github.com/yuh-zha/AlignScore
cd AlignScore

# Fix the classmethod bug in inference.py
echo "Patching AlignScore code to fix classmethod bug..."
sed -i 's/BERTAlignModel(model=model).load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)/BERTAlignModel.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, strict=False)/g' src/alignscore/inference.py

pip install . --no-deps
mkdir model
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt -P model
cd ../..

echo "Installing the package..."
pip install -e .

echo "Downloading NLP packages data..."
python -m spacy download en_core_web_sm
python3 -c "import nltk ; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

echo "Done!"
