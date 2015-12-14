#!/bin/bash

# list all the folders in the wsj directory
for i in $(ls ../data/wsj); do
	if [ "$i" != "MERGE.LOG" ]
		then
		$(mkdir ../data/wsj_parsed/$i)
		# list all the .mrg files
		for j in $(ls ../data/wsj/$i); do
			filename=$(basename "$j")
			filename="${filename%.*}"
			# convert the files to dependency tree and store them
			$(../converter/ptbconv.old/bin/p2d < ../data/wsj/$i/$j > ../data/wsj_parsed/$i/$filename)
		done
	fi
done