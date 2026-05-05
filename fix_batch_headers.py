from pathlib import Path
import csv

batch_dir = Path('batch')
rename_map = {'drug1': 'NSC1', 'drug2': 'NSC2', 'cell line': 'CELLNAME'}
files = sorted(batch_dir.glob('batch_*rows.csv'))
print(f'Found {len(files)} files')
for p in files:
    with p.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        new_header = [rename_map.get(c, c) for c in header]
        rows = [new_header] + [r for r in reader]
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f'Updated {p.name}')
