#!/bin/bash
echo "Fp1,Fp2,C3,C4,F3,Cz,F4,Fz,label,timestamp\n" > kek.csv
for f in data/*.csv
do
 echo "Processing $f"
 tail -n +2 $f >> kek.csv
 wc -l $f
done
wc -l kek.csv