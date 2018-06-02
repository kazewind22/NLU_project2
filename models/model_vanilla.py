import tensorflow as tf
from models import utils

class Model_vanilla:
    def __init__(self, args, textData):
        print('Creating single lstm Model')
        self.args = args
        self.textData = textData

        self.dropOutRate = None
        self.initial_state = None
        self.learning_rate = None
        self.loss = None
        self.optOp = None
        self.labels = None
        self.input = None
        self.target = None
        self.length = None
        self.embedded = None
        self.predictions = None
        self.batch_size = None
        self.corrects = None


        self.v0 = None
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.v4 = None
        self.v5 = None
        self.v6 = None
        self.v7 = None

        self.buildNetwork()

        x = tf.trainable_variables()
        print()

    def buildNetwork(self):
        with tf.name_scope('rnn'):
            # [batchSize, hiddenSize*2]
            out_vec_0, out_vec_1 = self.buildRNN()

        with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
            out_vec_0 = tf.reshape(out_vec_0, shape=[self.batch_size, self.args.hiddenSize*2])
            out_vec_1 = tf.reshape(out_vec_1, shape=[self.batch_size, self.args.hiddenSize*2])

            score_0 = utils.ffnn(inputs=out_vec_0, num_hidden_layers=self.args.ffnnLayers, hidden_size=self.args.ffnnSize,
                                 output_size=1, dropout=self.dropOutRate)
            score_1 = utils.ffnn(inputs=out_vec_1, num_hidden_layers=self.args.ffnnLayers, hidden_size=self.args.ffnnSize,
                                 output_size=1, dropout=self.dropOutRate)

            # [batchSize, 2]
            scores = tf.concat(values=[score_0, score_1], axis=-1, name='scores')
            # scores = tf.Print(input_=scores, data=[scores], message='scores')
        with tf.name_scope('predictions'):
            # [batchSize]
            # 0: the first candidate is the correct ending
            # 1: the 2nd candidate is the correct ending
            self.predictions = tf.argmax(scores, axis=-1, name='predictions', output_type=tf.int32)
            # single number
            self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self.labels, name='loss')
            # loss = tf.Print(loss, data=[loss])
            self.loss = tf.reduce_sum(loss)

        with tf.name_scope('backpropagation'):
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.apply_gradients(zip(gradients, trainable_params))

    def buildRNN(self):
        with tf.name_scope('placeholders'):
            # [batchSize, nSentences, maxSteps]
            input_shape = [None, self.args.nSentences, self.args.maxSteps]
            self.input = tf.placeholder(tf.int32, shape=input_shape, name='input')
            # [batchSize]
            self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
            # [batchSize, nSentences]
            self.length = tf.placeholder(tf.int32, shape=[None, self.args.nSentences], name='length')
            self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

            self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

        with tf.name_scope('embedding_layer'):
            if not self.args.preEmbedding:
                embeddings = tf.get_variable(
                    shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    name='embeddings')
            else:
                print('Using pretrained word embeddings!')
                embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)

            # [batchSize, nSentences, maxSteps, embeddingSize]
            self.embedded = tf.nn.embedding_lookup(embeddings, self.input)
            self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')

        with tf.name_scope('lstm'):
            with tf.variable_scope('cell', reuse=False):

                def get_cell(hiddenSize, dropOutRate):
                    cell = tf.contrib.rnn.LSTMCell(num_units=hiddenSize, state_is_tuple=True,
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
                                                             output_keep_prob=dropOutRate)
                    return cell

                # https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
                multiCell = []
                for i in range(self.args.rnnLayers):
                    multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
                multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)

            # [batchSize*nSentences, maxSteps, embeddingSize]
            self.embedded = tf.reshape(self.embedded, shape=[-1, self.args.maxSteps, self.args.embeddingSize])
            # [batchSize*nSentences]
            length = tf.reshape(self.length, shape=[-1])

            # [batchSize*nSentences, maxSteps, embeddingSize]
            outputs, state = tf.nn.dynamic_rnn(cell=multiCell, inputs=self.embedded, sequence_length=length,
                                               dtype=tf.float32)

            # [batchSize*nSentences, maxSteps]
            last_relevant_mask = tf.one_hot(indices=length-1, depth=self.args.maxSteps, name='last_relevant',
                                            dtype=tf.int32)
            # [batchSize*nSentences, hiddenSize]
            last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

        with tf.variable_scope('sentence_attention', reuse=tf.AUTO_REUSE):
            # [batchSize, nSentences, hiddenSize]
            outputs_vec = tf.reshape(last_relevant_outputs, shape=[-1, self.args.nSentences, self.args.hiddenSize], name='sentence_vecs')

            # [batchSize, (nSentences-2), hiddenSize]
            context_vec = tf.slice(outputs_vec, begin=[0, 0, 0], size=[-1, (self.args.nSentences-2), self.args.hiddenSize], name='context_vec')
            # [batchSize, (nSentences-2)*hiddenSize]
            context_vec = tf.reshape(context_vec, shape=[-1, (self.args.nSentences-2)*self.args.hiddenSize], name='context_vec')
            # [batchSize, hiddenSize]
            context_vec = utils.ffnn(inputs=context_vec, num_hidden_layers=self.args.pffnnLayers, hidden_size=self.args.pffnnSize,
                                 output_size=self.args.hiddenSize, dropout=self.dropOutRate)

            # [batchSize, 1, hiddenSize]
            candidate_0 = tf.slice(outputs_vec, begin=[0, self.args.nSentences-2, 0], size=[-1, 1, -1])
            candidate_1 = tf.slice(outputs_vec, begin=[0, self.args.nSentences-1, 0], size=[-1, 1, -1])

            # [batchSize, hiddenSize]
            candidate_0 = tf.squeeze(candidate_0, name='candidate_0')
            candidate_1 = tf.squeeze(candidate_1, name='candidate_1')


            # [batchSize, hiddenSize*2]
            out_vec_0 = tf.concat(values=[context_vec, candidate_0], axis=-1, name='out_vec_0')

            # [batchSize, hiddenSize*2]
            out_vec_1 = tf.concat(values=[context_vec, candidate_1], axis=-1, name='out_vec_0')

        return out_vec_0, out_vec_1

    @staticmethod
    def build_attention(from_vector, to_vector, to_vector_length, hiddenSize, attSize, to_vector_num):
        """
        Similar attention mechanism to https://arxiv.org/pdf/1409.0473.pdf

        :param from_vector: [batchSize, hiddenSize]
        :param to_vector: [batchSize, maxSteps, hiddenSize]
        :param to_vector_length: [batchSize]
        :param hiddenSize: [batchSize, maxSteps, hiddenSize]
        :param attSize: [batchSize]
        :param to_vector_num: [batchSize, maxSteps, hiddenSize]
        :return: att_vec: [batchSize, hiddenSize]
        """

        # [batchSize, 1, hiddenSize]
        from_vector = tf.expand_dims(from_vector, axis=1)

        # [batchSize, maxSteps, hiddenSize]
        from_vector = tf.tile(from_vector, multiples=[1, to_vector_num, 1])

        # [hiddenSize, attSize]
        weights_from = tf.get_variable(shape=[hiddenSize, attSize], name='weights_from')
        weights_to = tf.get_variable(shape=[hiddenSize, attSize], name='weights_to')

        # [batchSize*maxSteps, attSize]
        from_vector_ = tf.matmul(tf.reshape(from_vector, shape=[-1, hiddenSize]),
                                 weights_from, name='from_vector_')
        to_vector_ = tf.matmul(tf.reshape(to_vector, shape=[-1, hiddenSize]),
                               weights_to, name='to_vector_')

        # [batchSize*maxSteps, attSize]
        output_vec = tf.tanh(from_vector_ + to_vector_, name='output_vec')

        weights_project = tf.get_variable(shape=[attSize, 1], name='weights_project')

        # [batchSize*maxSteps, 1]
        logits = tf.matmul(output_vec, weights_project)

        # [batchSize*maxSteps]
        logits = tf.squeeze(logits, axis=-1)

        # [batchSize, maxSteps]
        logits = tf.reshape(logits, shape=[-1, to_vector_num], name='logits')
        # [batchSize, maxSteps]
        mask = tf.sequence_mask(lengths=to_vector_length, maxlen=to_vector_num, dtype=tf.float32)
        mask = tf.log(mask, name='mask')

        # [batchSize, maxSteps]
        logits_masked = tf.add(logits, mask, name='logits_masked')

        # [batchSize, maxSteps]
        # alpha is 0 for invalid time steps
        alpha = tf.nn.softmax(logits_masked, axis=-1, name='alpha')
        # [batchSize, maxSteps, 1]
        alpha = tf.expand_dims(alpha, axis=-1)

        # [batchSize, maxSteps, hiddenSize]
        alpha = tf.tile(alpha, multiples=[1, 1, hiddenSize])
        # multiply each time step with its corresponding weights
        # [batchSize, maxSteps, hiddenSize]
        att_vec = tf.multiply(to_vector, alpha)

        # [batchSize, hiddenSize]
        att_vec = tf.reduce_sum(att_vec, axis=1, name='att_vec')

        return att_vec

    def step(self, batch, test=False):
        feed_dict = {}

        # [batchSize, maxSteps]
        input_ = []
        length = []
        labels = []

        for sample in batch.samples:
            input_.append(sample.input_)
            labels.append(sample.label)
            length.append(sample.length)

        feed_dict[self.labels] = labels
        feed_dict[self.input] = input_
        feed_dict[self.length] = length
        feed_dict[self.batch_size] = len(length)

        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.predictions, self.corrects)
        else:
            # during test, do not use drop out!!!!
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.loss, self.predictions, self.corrects)

        return ops, feed_dict
