import glob
import re

values = []
for file_path in glob.glob('*.log'):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'GFlops geometric:\s*([\d.]+)', line)
            if match:
                try:
                    value = float(match.group(1))
                    values.append(value)
                except ValueError:
                    continue

if values:
    print(f'max: {max(values)}')
    print(f'min: {min(values)}')
print(len(values))