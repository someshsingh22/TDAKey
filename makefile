clean:
	black .
	isort . --profile black
	flake8 . --ignore=F401,E501,E402

install:
	pip install -r requirements.txt

clear:
	find . -name '__pycache__' | xargs rm -r -f
	find . -name 'DS_Store' | xargs rm -f
	rm logs/*.log
