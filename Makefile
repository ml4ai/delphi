all: test docs

docs:
	cd docs; make html

extensions: 
	mkdir -p build && \
	cd build  && cmake .. &&  cmake --build . -- -j8 DelphiPython && \
	cp *.so ../delphi/cpp

test: extensions
	pytest \
	  --cov-report term-missing:skip-covered --cov=delphi\
	  --doctest-modules\
	  --ignore=delphi/cpp\
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
