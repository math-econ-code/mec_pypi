from setuptools import setup, find_packages

setup(
	name="mec",
	version="v__0.0.1b",
	url="",
	authors=["Alfred Galichon"],
	author_email="ag133@nyu.edu",
	licence="GNU",
	python_requires=">= 3",
	packages=find_packages(),
    test_suite="mec.tests", 
	description="description of the package"	# can link markdown file here
)