docs:
	cd docs; make html

test:
	cd docs; make doctest
	pipenv run pytest --cov=delphi

pypi_upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
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
