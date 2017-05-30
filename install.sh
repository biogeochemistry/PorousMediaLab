#!/bin/bash
# installing script

echo "Compiling the script:"
python setup.py build_ext --inplace

echo "Removing temporary folders and files:"
rm -rfdv build
rm -v src/*.c
mv *.so lib/
