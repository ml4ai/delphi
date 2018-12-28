dev:
	pip install pipenv
	pipenv install -d --skip-lock

docs:
	cd docs; make html

requirements:
	pipenv run pipenv_to_requirements

test:
	pipenv run pytest \
	  -s\
	  --cov-report term:skip-covered --cov=delphi\
	  --doctest-module\
	  --ignore=delphi/program_analysis/pa_unit_tests \
	  --ignore=delphi/program_analysis/data\
	  --ignore=delphi/analysis/sensitivity/tests\
	  --ignore=delphi/translators/for2py/data\
	  delphi tests

pypi_upload:
	rm -rf dist
	pipenv run python setup.py sdist bdist_wheel
	twine upload dist/*

clean:
	rm -rf build dist
	rm *.json *.pkl *.csv

pkg_lock:
	pipenv lock
	pipenv run pipenv_to_requirements

push_test_data:
	tar -zcvf delphi_data.tgz  -C $(DELPHI_DATA)../ data
	scp delphi_data.tgz adarsh@vision.cs.arizona.edu:public_html/delphi_data.tgz
