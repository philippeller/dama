#rm -rf README_files
#jupyter nbconvert --to markdown --output-dir . notebooks/README.ipynb
sed -i -e 's/README_files/https:\/\/raw.githubusercontent.com\/philippeller\/dama\/master\/README_files/g' README.md
