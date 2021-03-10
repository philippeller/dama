# Increment version number in setup.py
rm -rf dist
python setup.py sdist
#twine upload dist/* -r pypitest
twine upload dist/* -r pypi
