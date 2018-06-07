import tensorflow as tf
import argparse
from models import utils
from models.textData import TextData
from models.model_att import Model_att
from models.model_vanilla import Model_vanilla
from models.model_satt import Model_satt
import os
import pickle as p
from tqdm import tqdm

class Predict:

    def __init__(self):
        self.args = None


        self.textData = None
        self.model = None
        self.outFile = None
        self.sess = None
        self.saver = None
        self.model_name = None
        self.model_path = None
        self.globalStep = 0
        self.summaryDir = None
        self.testOutFile = None
        self.summaryWriter = None
        self.mergedSummary = None

    @staticmethod
    def parse_args(args):

        parser = argparse.ArgumentParser()

        parser.add_argument('--resultDir', type=str, default='result', help='result directory')
        # data location
        dataArgs = parser.add_argument_group('Dataset options')

        dataArgs.add_argument('--random', type=int, default=3)
        dataArgs.add_argument('--backward', type=int, default=3)
        dataArgs.add_argument('--near', type=int, default=3)

        dataArgs.add_argument('--randomFile', type=str, default='random.csv')
        dataArgs.add_argument('--backwardFile', type=str, default='backward.csv')
        dataArgs.add_argument('--nearFile', type=str, default='near.csv')

        dataArgs.add_argument('--summaryDir', type=str, default='summaries')
        dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')

        dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
        dataArgs.add_argument('--trainFile', type=str, default='train_dummy.csv')
        dataArgs.add_argument('--valFile', type=str, default='val.csv')
        dataArgs.add_argument('--testFile', type=str, default='test.csv')

        dataArgs.add_argument('--embedFile', type=str, default='glove.840B.300d.txt')
        dataArgs.add_argument('--vocabSize', type=int, default=-1, help='vocab size, use the most frequent words')

        # neural network options
        nnArgs = parser.add_argument_group('Network options')
        nnArgs.add_argument('--embeddingSize', type=int, default=300)
        nnArgs.add_argument('--nSentences', type=int, default=6)
        nnArgs.add_argument('--hiddenSize', type=int, default=512, help='hiddenSize for RNN sentence encoder')
        nnArgs.add_argument('--attSize', type=int, default=512)
        nnArgs.add_argument('--sentenceAttSize', type=int, default=512)
        nnArgs.add_argument('--rnnLayers', type=int, default=1)
        nnArgs.add_argument('--maxSteps', type=int, default=30)
        nnArgs.add_argument('--numClasses', type=int, default=2)
        nnArgs.add_argument('--ffnnLayers', type=int, default=2)
        nnArgs.add_argument('--ffnnSize', type=int, default=300)
        nnArgs.add_argument('--pffnnLayers', type=int, default=2)
        nnArgs.add_argument('--pffnnSize', type=int, default=512)
        nnArgs.add_argument('--nn', type=str, default='att')
        # training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--dataProcess', action='store_true')
        trainingArgs.add_argument('--modelPath', type=str, default='saved')
        trainingArgs.add_argument('--preEmbedding', action='store_true')
        trainingArgs.add_argument('--dropOut', type=float, default=1.0, help='dropout rate for RNN (keep prob)')
        trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='learning rate')
        trainingArgs.add_argument('--batchSize', type=int, default=100, help='batch size')
        # max_grad_norm
        ## do not add dropOut in the test mode!
        trainingArgs.add_argument('--twitterTest', action='store_true', help='whether or not do test in twitter dataset')
        trainingArgs.add_argument('--epochs', type=int, default=200, help='most training epochs')
        trainingArgs.add_argument('--device', type=str, default='/gpu:0', help='use the first GPU as default')
        trainingArgs.add_argument('--loadModel', action='store_true', help='whether or not to use old models')
        return parser.parse_args(args)

    def main(self, args=None):
        print('Tensorflow version {}'.format(tf.VERSION))

        # initialize args
        self.args = self.parse_args(args)


        self.args.resultDir = self.args.nn +'_' + self.args.resultDir
        self.args.modelPath = self.args.nn +'_' + self.args.modelPath
        self.args.summaryDir = self.args.nn +'_' + self.args.summaryDir

        if not os.path.exists(self.args.resultDir):
            os.makedirs(self.args.resultDir)

        if not os.path.exists(self.args.modelPath):
            os.makedirs(self.args.modelPath)

        if not os.path.exists(self.args.summaryDir):
            os.makedirs(self.args.summaryDir)

        self.outFile = utils.constructFileName(self.args, prefix=self.args.resultDir)
        self.args.datasetName = utils.constructFileName(self.args, prefix=self.args.dataDir)
        datasetFileName = os.path.join(self.args.dataDir, self.args.datasetName)

        if not os.path.exists(datasetFileName):
            self.textData = TextData(self.args)
            with open(datasetFileName, 'wb') as datasetFile:
                p.dump(self.textData, datasetFile)
            print('dataset created and saved to {}'.format(datasetFileName))
        else:
            with open(datasetFileName, 'rb') as datasetFile:
                self.textData = p.load(datasetFile)
            print('dataset loaded from {}'.format(datasetFileName))

        if self.args.dataProcess:
            exit(0)

        sessConfig = tf.ConfigProto(allow_soft_placement=True)
        sessConfig.gpu_options.allow_growth = True

        self.model_path = utils.constructFileName(self.args, prefix=self.args.modelPath, tag='model')
        self.model_name = os.path.join(self.model_path, 'model')

        self.sess = tf.Session(config=sessConfig)
        # summary writer
        self.summaryDir = utils.constructFileName(self.args, prefix=self.args.summaryDir)

        with tf.device(self.args.device):
            if self.args.nn == 'vanilla':
                print('Creating vanilla model!')
                self.model = Model_vanilla(self.args, self.textData)
            elif self.args.nn == 'att':
                print('Creating model with sentences and words attention!')
                self.model = Model_att(self.args, self.textData)
            else:
                print('Creating model with only words attention!')
                self.model = Model_satt(self.args, self.textData)
            print('Model created')

            # saver can only be created after we have the model
            self.saver = tf.train.Saver()

            self.summaryWriter = tf.summary.FileWriter(self.summaryDir, self.sess.graph)
            self.mergedSummary = tf.summary.merge_all()

            if self.args.loadModel:
                # load model from disk
                if not os.path.exists(self.model_path):
                    print('model does not exist on disk!')
                    print(self.model_path)
                    exit(-1)

                self.saver.restore(sess=self.sess, save_path=self.model_name)
                print('Variables loaded from disk {}'.format(self.model_name))
            else:
                init = tf.global_variables_initializer()
                # initialize all global variables
                self.sess.run(init)
                print('All variables initialized')

            self.train(self.sess)

    def train(self, sess):
        print('Start training')

        out = open(self.outFile, 'w', 1)
        out.write(self.outFile + '\n')
        utils.writeInfo(out, self.args)

        current_valAcc = 0.0

        for e in range(self.args.epochs):
            # training
            #trainBatches = self.textData.train_batches
            trainBatches = self.textData.get_batches(tag='train')
            totalTrainLoss = 0.0

            # cnt of batches
            cnt = 0

            total_samples = 0
            total_corrects = 0
            for nextBatch in tqdm(trainBatches):
                cnt += 1
                self.globalStep += 1

                total_samples += nextBatch.batch_size
                ops, feed_dict = self.model.step(nextBatch, test=False)

                _, loss, predictions, corrects = sess.run(ops, feed_dict)
                total_corrects += corrects
                totalTrainLoss += loss

                # average across samples in this step
                self.summaryWriter.add_summary(utils.makeSummary({"trainLoss": loss}), self.globalStep)
            # compute perplexity over all samples in an epoch
            trainAcc = total_corrects*1.0/total_samples
            print('\nepoch = {}, Train, loss = {}, trainAcc = {}'.
                  format(e, totalTrainLoss, trainAcc))
            out.write('\nepoch = {}, loss = {}, trainAcc = {}\n'.
                  format(e, totalTrainLoss, trainAcc))
            out.flush()
            valAcc, valLoss = self.test(sess, tag='val')

            print('Val, loss = {}, valAcc = {}'.
                  format(valLoss, valAcc))
            out.write('Val, loss = {}, valAcc = {}\n'.
                  format(valLoss, valAcc))

            testAcc, testLoss = self.test(sess, tag='test')
            print('Test, loss = {}, testAcc = {}'.
                  format(testLoss, testAcc))
            out.write('Test, loss = {}, testAcc = {}\n'.
                  format(testLoss, testAcc))

            out.flush()

            # we do not use cross val currently, just train, then evaluate
            if valAcc >= current_valAcc:
                current_valAcc = valAcc
                print('New valAcc {} at epoch {}'.format(valAcc, e))
                out.write('New valAcc {} at epoch {}\n'.format(valAcc, e))
                save_path = self.saver.save(sess, save_path=self.model_name)
                print('model saved at {}'.format(save_path))
                out.write('model saved at {}\n'.format(save_path))

            out.flush()
        out.close()


    def test(self, sess, tag = 'val'):
        if tag == 'val':
            print('Validating\n')
            batches = self.textData.val_batches
        else:
            print('Testing\n')
            batches = self.textData.test_batches

        cnt = 0

        total_samples = 0
        total_corrects = 0
        total_loss = 0.0
        all_predictions = []
        for idx, nextBatch in enumerate(tqdm(batches)):
            cnt += 1

            total_samples += nextBatch.batch_size
            ops, feed_dict = self.model.step(nextBatch, test=True)

            loss, predictions, corrects = sess.run(ops, feed_dict)
            all_predictions.extend(predictions)
            total_loss += loss
            total_corrects += corrects

        acc = total_corrects*1.0/total_samples
        return acc, total_loss


