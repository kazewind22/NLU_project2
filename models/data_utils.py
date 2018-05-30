class Sample:
    def __init__(self, input_, sentences, length, label):
        self.input_ = input_

        # words are actually not useful, just for debug
        self.sentences = sentences

        # list of shape [5], length of each sentence in the model
        self.length = length

        # 0 or 1, whether the 5th sentence is the correct ending
        self.label = label

class Batch:
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)
