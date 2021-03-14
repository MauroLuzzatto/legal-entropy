.DEFAULT: build


init:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m spacy download de_core_news_sm

black:
	python -m black src

activate:
	conda activate entropy_env

env.export:
	conda env export > environment.yml

env.install:
	conda env create -f environment.yml

