from random import shuffle

with open("data_churn_total.csv", "r") as f:
    header, *lines = f.readlines()

print("Total lines:", len(lines))
shuffle(lines)

EXTRACT_LENGTH = 43
EXTRACT_PRED_LENGTH = 50
lines_extract = lines[:EXTRACT_LENGTH]
lines_extract_pred = lines[EXTRACT_LENGTH:EXTRACT_LENGTH + EXTRACT_PRED_LENGTH]
lines = lines[EXTRACT_LENGTH + EXTRACT_PRED_LENGTH:]

def get_index(line):
    return int(line.split(",")[0])

lines_extract.sort(key=get_index)
lines_extract_pred.sort(key=get_index)
lines.sort(key=get_index)

print("Extract lines:", len(lines_extract))
print("Extract lines for prediction only:", len(lines_extract_pred))
print("Leftover lines:", len(lines))

with open("data_churn_extract.csv", "w") as f:
    f.writelines([header] + lines_extract)

with open("data_churn_extract_pred.csv", "w") as f:
    def remove_churn(line):
        return line.rsplit(",", 1)[0] + "\n"
    f.writelines([remove_churn(header)] + list(map(remove_churn, lines_extract_pred)))

with open("data_churn.csv", "w") as f:
    f.writelines([header] + lines)
