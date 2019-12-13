#!/bin/bash
echo "Fp1,Fp2,C3,C4,F3,Cz,F4,Fz,label,timestamp\n" > all.csv
for f in data/*.csv
do
 echo "Processing $f"
 tail -n +2 $f >> all.csv
 wc -l $f
done
wc -l all.csv