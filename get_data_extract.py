from random import shuffle

with open("data_churn_total.csv", "r") as f:
    header, *lines = f.readlines()

print("Total lines:", len(lines))
shuffle(lines)

EXTRACT_LENGTH = 43
lines_extract = lines[:EXTRACT_LENGTH]
lines = lines[EXTRACT_LENGTH:]

def get_index(line):
    return int(line.split(",")[0])

lines_extract.sort(key=get_index)
lines.sort(key=get_index)

print("Extract lines:", len(lines_extract))
print("Leftover lines:", len(lines))

with open("data_churn_extract.csv", "w") as f:
    f.writelines([header] + lines_extract)

with open("data_churn.csv", "w") as f:
    f.writelines([header] + lines)
