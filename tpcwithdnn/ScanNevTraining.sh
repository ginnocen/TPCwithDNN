#!/bin/bash

Nev_arr=(100 200 300 400 500 600 700 800 900);

for i in "${!Nev_arr[@]}" ; do
	echo "python steer_analysis.py --Nev ${Nev_arr[$i]}";
	python steer_analysis.py --Nev ${Nev_arr[$i]};
done

echo "Done!";
