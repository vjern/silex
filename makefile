.ONESHELL:

testx:
	python -m pytest tests -vv

test:
	python -m pytest --cov=g.py --cov-report term-missing g.py -vvv

lint:
	# git ls-files for local files
	git ls-tree --full-tree -r --name-only HEAD \
	| grep ".py$$" \
	| xargs flake8 --max-line-length=100
