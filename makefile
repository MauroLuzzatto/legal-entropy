.DEFAULT: build


init:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m spacy download de_core_news_sm

black:
	python -m black src
