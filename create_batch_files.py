import csv
import os
import random

src = 'datasets/train_500_rows.csv'
out_dir = 'batch'
os.makedirs(out_dir, exist_ok=True)

with open(src, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

if not rows:
    raise SystemExit('Dataset is empty')

header = rows[0]
data = rows[1:]
if len(data) < 80:
    raise SystemExit(f'Not enough rows: {len(data)}')

for n in range(10, 90, 10):
    sample = random.sample(data, n)
    out_path = os.path.join(out_dir, f'batch_{n}rows.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(sample)
    print(f'Created {out_path} with {n} rows')
