docs:
	cd docs; make html

test:
	cd docs; make doctest
	pipenv run pytest delphi/tests

pypi_upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/*

clean:
	rm -rf build dist

pkg_lock:
	pipenv lock
	pipenv run pipenv_to_requirements
