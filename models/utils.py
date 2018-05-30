import os
import tensorflow as tf

def makeSummary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 2:
        current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
        current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.matmul(current_inputs, output_weights) + output_bias

    return outputs

def constructFileName(args, prefix=None, tag=None):

    if prefix == args.dataDir:
        file_name = ''
        file_name += prefix + '-'
        file_name += str(args.vocabSize) + '-'
        file_name += str(args.batchSize) + '-'
        file_name += str(args.maxSteps) + '.pkl'
        return file_name

    file_name = ''
    file_name += 'embeddingSize_' + str(args.embeddingSize)
    file_name += '_hiddenSize_' + str(args.hiddenSize)
    file_name += '_rnnLayers_' + str(args.rnnLayers)
    file_name += '_maxSteps_' + str(args.maxSteps)
    file_name += '_dropOut_' + str(args.dropOut)

    file_name += '_learningRate_' + str(args.learningRate)
    file_name += '_batchSize_' + str(args.batchSize)
    file_name += '_vocabSize_' + str(args.vocabSize)
    file_name += '_preEmbedding_' + str(args.preEmbedding)
    file_name += '_attSize_' + str(args.attSize)
    if tag != 'model':
        file_name += '_loadModel_' + str(args.loadModel)

    file_name = os.path.join(prefix, file_name)

    return file_name

def writeInfo(out, args):
    out.write('embeddingSize {}\n'.format(args.embeddingSize))
    out.write('hiddenSize {}\n'.format(args.hiddenSize))
    out.write('attSize {}\n'.format(args.attSize))
    out.write('rnnLayers {}\n'.format(args.rnnLayers))

    out.write('maxSteps {}\n'.format(args.maxSteps))
    out.write('dropOut {}\n'.format(args.dropOut))

    out.write('learningRate {}\n'.format(args.learningRate))
    out.write('batchSize {}\n'.format(args.batchSize))
    out.write('epochs {}\n'.format(args.epochs))

    out.write('loadModel {}\n'.format(args.loadModel))

    out.write('vocabSize {}\n'.format(args.vocabSize))
    out.write('preEmbeddings {}\n'.format(args.preEmbedding))

