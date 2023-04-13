pip install nltk
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

python -m spacy download en_core_web_md


uvicorn similarity:app --port 9000 --reload --log-level info
python3.10 -m uvicorn similarity:app --port 9000 --reload --log-level info