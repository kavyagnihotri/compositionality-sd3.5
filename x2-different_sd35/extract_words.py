import json

# Input and output file paths
parent_dir = '/mnt/lustre/work/hein/ndsingh40/datasets/clicWords'
json_file_path = '.json'
output_file_path = 'words.txt'

# Set to store unique words (to avoid duplicates)
words = set()

# Read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    # Read each line as a separate JSON object
    for line in file:
        try:
            # Parse the JSON object
            data = json.loads(line.strip())
            # Extract the key (the main word)
            for key in data:
                words.add(key)
                # Extract the values (related words)
                for value in data[key]:
                    words.add(value)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON line: {line.strip()} - {e}")

# Write words to a text file, each on a new line
with open(output_file_path, 'w', encoding='utf-8') as file:
    for word in sorted(words):  # Sorting for consistent output
        file.write(word + '\n')

print(f"Words have been written to {output_file_path}")