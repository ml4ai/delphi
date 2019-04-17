#!/bin/bash
# A shell script to read file line by line
 
filename="all-fortran-tests/file-list.txt"
 
while read line
do
    # $line variable contains current line read from the file
    # display $line text on the screen or do something with it.
 
    echo "$line"
    ./autoTranslate all-fortran-tests/$line
done < $filename
