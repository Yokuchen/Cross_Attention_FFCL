import csv


# Function to convert the CSV file
def convert_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        # CSV reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        writer.writerow(['id', 'text', 'label'])

        # Iterate over each row in the input file
        for id, row in enumerate(reader, start=1):
            # Extract label and text, ignoring extra commas
            if row[0] == 'ham':
                raw_label = 0
            elif row[0] == 'spam':
                raw_label = 1
            label, text = raw_label, row[1]
            # Write the new format [id, text, label]
            writer.writerow([id, text, label])
