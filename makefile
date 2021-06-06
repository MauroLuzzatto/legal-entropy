.DEFAULT: build


setup: env.create activate init


init:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m spacy download de_core_news_sm

black:
	python -m black src

activate:
	conda activate entropy_env

env.create:
	conda create --name entropy_env python=3.8.5

env.export:
	conda env export > environment.yml

env.install:
	conda env create -f environment.yml

