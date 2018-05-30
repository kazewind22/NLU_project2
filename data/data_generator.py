import csv
import random
import numpy as np

def generate_data(data_train, data_out):
    random.seed(22)
    header = np.array(['InputStoryid','InputSentence1','InputSentence2','InputSentence3',
        'InputSentence4','RandomFifthSentenceQuiz1','RandomFifthSentenceQuiz2','AnswerRightEnding'])

    # shuffle correct ending to get random ending
    corpus = []
    with open(data_train, 'r') as inputfile:
        reader = csv.reader(inputfile)
        reader.__next__() #drop header
        for row in reader:
            corpus.append(row[6])
    random.shuffle(corpus)

    def process_row(row, neg):
        if random.randint(0,1):
            row.append(row[5])
            row[5] = neg
            row.append(2)
        else:
            row.append(neg)
            row.append(1)
        return row

    with open(data_train, 'r') as inputfile, open(data_out, 'w') as outputfile:
        reader = csv.reader(inputfile)
        writer = csv.writer(outputfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reader.__next__()
        writer.writerow(header)
        for i, row in enumerate(reader):
            # remove story title
            row.remove(row[1])

            # write random negative ending
            rand = process_row(row.copy(), corpus[i])
            writer.writerow(rand)

            # write backward negative ending
            back = process_row(row.copy(), row[random.randint(1,4)])
            writer.writerow(back)


if __name__ == "__main__":
    data_train = "./train_stories.csv"
    data_out   = "./train_stories_neg.csv"
    generate_data(data_train, data_out)
