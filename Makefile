docs:
	cd docs; make html

extensions: 
	cd delphi/cpp; cmake .; make -j

test: extensions
	time pytest \
	  --cov-report term-missing:skip-covered --cov=delphi\
	  --doctest-module\
	  --ignore=delphi/analysis/sensitivity/tests\
	  --ignore=delphi/cpp/pybind11\
	  --ignore=delphi/cpp/nlohmann\
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
