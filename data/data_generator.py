import csv
import random
import numpy as np

def generate_data_random(data_train):
    random.seed(22)
    header = np.array(['InputSentence1','InputSentence2','InputSentence3',
        'InputSentence4','AnswerRightEnding','FakeEnding1','FakeEnding2',
        'FakeEnding3','FakeEnding4','FakeEnding5','FakeEnding6'])

    # shuffle correct ending to get random ending
    corpus = []
    with open(data_train, 'r') as inputfile:
        reader = csv.reader(inputfile)
        reader.__next__() #drop header
        for row in reader:
            corpus.append(row[6])

    with open(data_train, 'r') as inputfile, open(data_train[:-4]+'_random.csv', 'w') as outputfile:
        reader = csv.reader(inputfile)
        writer = csv.writer(outputfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reader.__next__()
        writer.writerow(header)
        for row in reader:
            # remove story id
            row.remove(row[0])
            # remove story title
            row.remove(row[0])

            # write random negative ending
            i = 0
            while i < 6:
                sample = random.choice(corpus)
                if sample == row[4]:
                    pass
                else:
                    row.append(sample)
                    i += 1
            writer.writerow(row)

def generate_data_back(data_train):
    random.seed(22)
    header = np.array(['InputSentence1','InputSentence2','InputSentence3',
        'InputSentence4','AnswerRightEnding','FakeEnding1','FakeEnding2',
        'FakeEnding3','FakeEnding4'])

    with open(data_train, 'r') as inputfile, open(data_train[:-4]+'_back.csv', 'w') as outputfile:
        reader = csv.reader(inputfile)
        writer = csv.writer(outputfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reader.__next__()
        writer.writerow(header)
        for row in reader:
            # remove story id
            row.remove(row[0])
            # remove story title
            row.remove(row[0])

            row += row[0:4]
            writer.writerow(row)


if __name__ == "__main__":
    data_train = "./train_stories.csv"
    generate_data_random(data_train)
    generate_data_back(data_train)
