import sys

f = open(sys.argv[1])
lines = [line.strip().split(',') for line in f.readlines()]
f.close()

n_fields = len(lines[0])
fieldnames = lines[0]
fieldnames_split = []
rows = lines[1:]
field_widths = [0 for i in range(n_fields)]

for row in rows:
    for i in range(n_fields):
        if len(row[i]) > field_widths[i]:
            field_widths[i] = len(row[i])

field_widths = [max(field_widths[i], 8) for i in range(n_fields)]

for i, fieldname in enumerate(fieldnames):
    fieldname_split = [fieldname[j * field_widths[i]:(j + 1) * field_widths[i]] for j in range(len(fieldname) // field_widths[i] + 1)]
    fieldname_split = fieldname_split[:-1] if len(fieldname_split[-1]) == 0 else fieldname_split
    fieldname_split[-1] += ' ' * (field_widths[i] - len(fieldname_split[-1]))
    fieldnames_split.append(fieldname_split)

for i in range(max([len(fieldname_split) for fieldname_split in fieldnames_split])):
    for j in range(n_fields):
        if i < len(fieldnames_split[j]):
            print(fieldnames_split[j][i], end = ' ')
        else:
            print(' ' * field_widths[j], end = ' ')
    print()

for row in rows:
    for i in range(n_fields):
        print(f'{row[i]}{" " * (field_widths[i] - len(row[i]))} ', end = '')
    print()

for i in range(max([len(fieldname_split) for fieldname_split in fieldnames_split])):
    for j in range(n_fields):
        if i < len(fieldnames_split[j]):
            print(fieldnames_split[j][i], end = ' ')
        else:
            print(' ' * field_widths[j], end = ' ')
    print()