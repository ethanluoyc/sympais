#!/bin/bash


# add the files corresponding to the benchmarks you desire to run here.

FILES="
normal/framingham-hypten-0.m
normal/example-ckd-epi-0.m
normal/example-ckd-epi-1.m
normal/example-ckd-epi-simple-0.m
normal/example-ckd-epi-simple-1.m
normal/example-invPend-0.m
normal/framingham-0.m
normal/framingham-1.m
normal/framingham-2.m
normal/framingham-hypten-3.m
"

mkdir -p ../out/

for FILE in $FILES
do
	BASENAME=$(basename $FILE)
	echo "$FILE"
	math < $FILE > "../out/${BASENAME}.out"
done
