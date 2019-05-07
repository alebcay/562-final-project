#!/usr/bin/env bash

set -x
set -e

realEntries=$(wc -l real_labels.csv | awk '{ print $1 }')
cartoonEntries=$(wc -l cartoon_labels.csv | awk '{ print $1 }')

dd if=/dev/random of=cartoonrand count=1024
dd if=/dev/random of=realrand count=1024

shuf --random-source=cartoonrand cartoon_labels.csv > cartoon_labels_shuffled.csv
shuf --random-source=cartoonrand cartoon_landmarks.csv > cartoon_landmarks_shuffled.csv

shuf --random-source=realrand real_labels.csv > real_labels_shuffled.csv
shuf --random-source=realrand real_landmarks.csv > real_landmarks_shuffled.csv

rm cartoonrand realrand

if [ "$realEntries" -lt "$cartoonEntries" ]; then
	head -$realEntries cartoon_labels_shuffled.csv > combined_labels.csv
	head -$realEntries cartoon_landmarks_shuffled.csv > combined_landmarks.csv
	cat real_labels_shuffled.csv >> combined_labels.csv
	cat real_landmarks_shuffled.csv >> combined_landmarks.csv
elif [ "$cartoonEntries" -lt "$realEntries" ]; then
	head -$cartoonEntries real_labels_shuffled.csv > combined_labels.csv
	head -$cartoonEntries real_landmarks_shuffled.csv > combined_landmarks.csv
	cat cartoon_labels_shuffled.csv >> combined_labels.csv
	cat cartoon_landmarks_shuffled.csv >> combined_landmarks.csv
else
	cp cartoon_labels_shuffled.csv combined_labels.csv
	cp cartoon_landmarks_shuffled.csv combined_landmarks.csv
	cat real_labels_shuffled.csv >> combined_labels.csv
	cat real_landmarks_shuffled.csv >> combined_landmarks.csv
fi

combinedEntries=$(wc -l combined_labels.csv | awk '{ print $1 }')

let "realSplit = $realEntries * 3 / 4"
let "cartoonSplit = $cartoonEntries * 3 / 4"
let "combinedSplit = $combinedEntries * 3 / 4"

split -l $realSplit real_labels_shuffled.csv
mv xaa real_labels_train.csv
mv xab real_labels_test.csv

split -l $realSplit real_landmarks_shuffled.csv
mv xaa real_landmarks_train.csv
mv xab real_landmarks_test.csv

split -l $cartoonSplit cartoon_labels_shuffled.csv
mv xaa cartoon_labels_train.csv
mv xab cartoon_labels_test.csv

split -l $cartoonSplit cartoon_landmarks_shuffled.csv
mv xaa cartoon_landmarks_train.csv
mv xab cartoon_landmarks_test.csv

split -l $combinedSplit combined_labels.csv
mv xaa combined_labels_train.csv
mv xab combined_labels_test.csv

split -l $combinedSplit combined_landmarks.csv
mv xaa combined_landmarks_train.csv
mv xab combined_landmarks_test.csv