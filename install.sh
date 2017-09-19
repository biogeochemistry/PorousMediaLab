#!/bin/bash
# installing script

echo "Compiling the script:"
python с_setup.py build_ext --inplace

echo "Removing temporary folders and files:"
rm -rfdv build
rm -v porousmedialab/*.c
mv *.so lib/
