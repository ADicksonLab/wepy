all:

check:
	python -m CoverageTestRunner
	rm -f .coverage

clean:
	rm -f *.pyc *.pyo .coverage
