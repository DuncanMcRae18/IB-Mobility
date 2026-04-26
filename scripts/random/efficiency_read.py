import yaml
import csv

with open("/home/duncan/Documents/School/Current Semester/Physics Project/Main/test/Eg_1.67/feincs2019_pint10_1/optimizer_summary.yaml") as stream:
    opt = yaml.unsafe_load(stream)

with open("../optimizer_summary.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Key", "Value"])
    for key, value in opt.items():
        writer.writerow([key, str(value)])
        print(f"{key}: {value}")