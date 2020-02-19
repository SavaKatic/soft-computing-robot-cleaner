import csv

if __name__ == '__main__':

    rows = []
    with open('test_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                rows.append(row)
                line_count += 1

    with open('test_labels2.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

        for row in rows:
            if row[0] and row[1] and row[2] and row[3] and row[4] and row[5] and row[6] and row[7]:
                writer.writerow(row)

    print("Transfer successful!")

