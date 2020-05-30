import sys

if len(sys.argv) != 2:
    print('Usage: python official_evaluation.py <path to tsv file>')

file_name = open(sys.argv[1], 'r')

precisions = []
recalls = []

eof = False
while True:
    sentence_word = []
    sentence_real = []
    sentence_predicted = []
    while True:
        line = file_name.readline()
        if not line:
            eof = True
            break
        elif line == '\n':
            break
        line = line.split()
        sentence_word.append(line[0])
        sentence_real.append(line[1])
        sentence_predicted.append(line[2])
    
    i = 0
    bound = len(sentence_predicted)
    while i < bound:
        if sentence_predicted[i] == '1':
            start = i
            finish = i+1
            predicted_span_length = len(sentence_word[start])
            while finish < bound and sentence_predicted[finish] == '1':
                predicted_span_length += len(sentence_word[finish]) + 1
                finish += 1
            # span je [start, finish>
            real_span_length = 0
            if sentence_real[start] == '1':
                real_span_length += len(sentence_word[start])
            for j in range(start+1,finish):
                if sentence_real[j] == '1':
                    real_span_length += len(sentence_word[j])
                    if sentence_real[j-1] == '1':
                        real_span_length += 1
            i = finish
            precisions.append(real_span_length / predicted_span_length)
        else:
            i += 1 
    
    i = 0
    bound = len(sentence_real)
    while i < bound:
        if sentence_real[i] == '1':
            start = i
            finish = i+1
            real_span_length = len(sentence_word[start])
            while finish < bound and sentence_real[finish] == '1':
                real_span_length += len(sentence_word[finish]) + 1
                finish += 1
            # span je [start, finish>
            predicted_span_length = 0
            if sentence_predicted[start] == '1':
                predicted_span_length += len(sentence_word[start])
            for j in range(start+1,finish):
                if sentence_predicted[j] == '1':
                    predicted_span_length += len(sentence_word[j])
                    if sentence_predicted[j-1] == '1':
                        predicted_span_length += 1
            i = finish
            recalls.append(predicted_span_length / real_span_length)
        else:
            i += 1


    if eof:
        break    
        
P = sum(precisions) / len(precisions) if len(precisions) > 0 else 1
R = sum(recalls) / len(recalls) if len(recalls) > 0 else 1
F1 = (2*P*R) / (P + R)
print(P)
print(R)
print(F1)                

