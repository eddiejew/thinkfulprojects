"""combine_data.py"""

import csv
import string

# HAR dataset provides training and testing data
# However, we don't want to use their splits - we'd 
# like to create our own splits. We'll first have to 
# combine the UCI provided datasets into a single 
# file

# first clean column names from the feature.txt file
# rules for cleaning column names:
# remove parentheses, remove dashes, replace commas with underscores
def clean_string(s):
	translations = string.maketrans(",", "_")
	deletions = "()-"
	return s.translate(translations, deletions)

# clean column names and write to output file
def clean_column_names(fname_in, fname_out):
	SPACE = " "
	with open(fname_in, "rb") as f_in, open(fname_out, "wb") as f_out:
		reader = csv.reader(f_in, delimiter=SPACE)
		writer = csv.writer(f_out, delimiter=SPACE)
		for line in reader:
			writer.writerow([line[0], clean_string(line[1])])

def main():
	in_file = "./UCI HAR Dataset/features.txt"
	out_file = "./features_clean.txt"
	clean_column_names(in_file, out_file)

if __name__ == '__main__':
	main()

X_train = open("./UCI HAR Dataset/train/X_train.txt", "rb").read().splitlines()
y_train = open("./UCI HAR Dataset/train/y_train.txt", "rb").read().splitlines()
subject_train = open("./UCI HAR Dataset/train/subject_train.txt", "rb").read().splitlines()

X_test = open("./UCI HAR Dataset/test/X_test.txt", "rb").read().splitlines()
y_test = open("./UCI HAR Dataset/test/y_test.txt", "rb").read().splitlines()
subject_test = open("./UCI HAR Dataset/test/subject_test.txt", "rb").read().splitlines()

X_all = X_train + X_test
y_all = y_train + y_test
subject_all = subject_train + subject_test

with open("./X_all.txt", "wb") as Xf, \
         open("./y_all.txt", "wb") as yf, \
         open("./subject_all.txt", "wb") as sf:
     X_writer = csv.writer(Xf)
     y_writer = csv.writer(yf)
     subject_writer = csv.writer(sf)

     for xrow, yrow, srow in zip(X_all, y_all, subject_all):
     	X_writer.writerow([xrow])
     	y_writer.writerow([yrow])
     	subject_writer.writerow([srow])