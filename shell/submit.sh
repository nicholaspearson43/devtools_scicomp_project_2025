#!/bin/bash

REPO="devtools_scicomp_project_2025"
DESC="Descrizione"
LIC="Apache 2.0"
NAME="Nicholas Pearson"
email="nicholaspr1997"

conda create --name devtools_scicomp python=3.9
conda activate devtools_scicomp

python -m pip install pytest

git clone git@github.com:nicholaspearson43/${REPO}.git
cd ${REPO}

touch README.md
echo "Hello" >> README.md

git add .
git commit -m "first commit"
git push origin HEAD:main

mkdir src/pyclassify/
mkdir scripts
mkdir test
mkdir shell
mkdir experiments

touch src/pyclassify/__init__.py
touch src/pyclassify/utils.py
touch scripts/run.py
touch shell/submit.batch
touch shell/submit.sh
touch experiments/config.yaml
touch test/test_.py

python -m pip freeze > requirements.txt

wget 'https://github.com/dario-coscia/devtools_scicomp_project_2025/blob/main/pyproject.toml' -O pyproject.toml
sed -i 's/description = "INSERT"/description = "${DESC}"/' pyproject.toml
sed -i 's/file = "LICENSE"/file = ${LIC}/' pyproject.toml
sed -i 's/name = "INSERT"/name = "${NAME}"/' pyproject.toml
sed -i 's/email = "INSERT@gmail.com"/email = "{$EMAIL}@gmail.com/' pyproject.toml



echo "#Ignore .dat file and .data files" >> .gitignore
echo ".dat" >> .gitignore
echo ".data" >> .gitignore

git add .
git commit -m "structuring the package"
git push origin HEAD:main
