.PHONY: docs

docs:
	cd docs; make html

test:
	pytest \
	  -s\
	  --cov-report term-missing:skip-covered --cov=delphi\
	  --doctest-module\
	  --ignore=delphi/program_analysis/pa_unit_tests \
	  --ignore=delphi/program_analysis/data\
	  --ignore=delphi/analysis/sensitivity/tests\
	  --ignore=delphi/translators/for2py/data\
	  --ignore=tests/data\
	  delphi tests

pypi_upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/*

clean:
	rm -rf build dist
	rm *.json *.pkl *.csv

push_test_data:
	scp delphi.db vision:public_html
