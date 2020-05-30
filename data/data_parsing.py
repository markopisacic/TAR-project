import os
import re

ARTICLES_DIR = './articles/'
LABELS_DIR = './labels/'
OUT_DIR = './labeled_articles/'


def split_with_offsets(line):
    words = line.split()
    offsets = []
    running_offset = 0
    for word in words:
        word_offset = line.index(word, running_offset)
        word_len = len(word)
        running_offset = word_offset + word_len
        offsets.append((word, word_offset, running_offset - 1))
    return offsets


def is_in_span(spans, pos):
    for span in spans:
        if span[0] <= pos < span[1]:
            return 1
    return 0


def filter_doubles(l):
    n = []
    for i in range(len(l) - 1):
        if l[i] != l[i + 1] - 1:
            n.append(l[i])
    n.append(l[-1])
    return n


for doc_idx, filename in enumerate(os.listdir(LABELS_DIR)):
    print('Parsing doc ', str(doc_idx), filename)
    spans = []
    with open(LABELS_DIR + '/' + filename, encoding='utf-8') as file:
        for line in file:
            spans.append((int(line.split()[1]), int(line.split()[2])))
    spans = sorted(spans, key=lambda x: x[0])

    with open(ARTICLES_DIR + '/' + filename, encoding='utf-8') as file:
        text = file.read()
    labels = []
    for w, o1, o2 in split_with_offsets(text):
        labels.append((w, o1, is_in_span(spans, (o1 + o2) / 2)))

    newlines = [m.start() for m in re.finditer('\n', text)]
    newlines = filter_doubles(newlines)
    if newlines[-1] != len(text):
        newlines.append(len(text))

    out_file = open(OUT_DIR + '/' + filename, "w+", encoding='utf-8')
    current_newline_idx = 0
    for w, o, l in labels:
        if o > newlines[current_newline_idx]:
            current_newline_idx = current_newline_idx + 1 if current_newline_idx < len(newlines) - 1 else len(
                newlines) - 1
            out_file.write('\n')
        out_file.write(w + " " + str(l) + " " + str(filename) + " " + str(current_newline_idx) + "\n")
    out_file.close()

print('Done.')
