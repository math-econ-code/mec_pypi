from setuptools import setup, find_packages

setup(
	name="mec",
	version="0.0.3.1",
	url="",
	authors=["Alfred Galichon"],
	author_email="ag133@nyu.edu",
	licence="",
	python_requires=">= 3",
    install_requires=["gurobipy"],
	packages=find_packages(),
    test_suite="mec.tests", 
	description="description of the package"	# can link markdown file here 
)