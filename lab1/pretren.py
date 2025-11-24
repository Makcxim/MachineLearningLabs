import csv

input_file = '../globalterrorismdb_0718dist.csv'
output_file = 'gtd_clean.csv'


with open(input_file, mode='r', encoding='ISO-8859-1') as f_in, \
        open(output_file, mode='w', encoding='utf-8', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    next(reader)

    # Год, Регион, Убитые, Раненые

    count = 0
    for row in reader:
        try:
            # 1=Year, 10=Region, 98=nkill, 101=nwound
            year = row[1]
            region = row[10]

            kill = row[98] if row[98] != '' else '0'
            wound = row[101] if row[101] != '' else '0'

            writer.writerow([year, region, kill, wound])
            count += 1
        except IndexError:
            continue

print(f"{count} строк")
