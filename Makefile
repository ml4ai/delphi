test:
	cd docs; make doctest

pypi_upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/*
