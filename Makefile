all: test docs

docs:
	cd docs; make html

extensions: 
	mkdir -p build && \
	cd build  && cmake .. &&  cmake --build . -- -j2 DelphiPython && \
	cp *.so ../delphi/cpp

test: extensions
	time pytest --cov-report term-missing:skip-covered --cov=delphi \
	--ignore=tests/data tests

test_wm: extensions
	pytest -s tests/wm/test_cpp_evaluation.py
	pytest -s tests/wm/test_cpp_extensions.py
	pytest -s tests/wm/test_rest_api.py


pypi_upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/*

clean:
	rm -rf build dist
	rm -f *.json *.pkl *.csv
	mkdir build
