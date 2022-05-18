"""Main program for babble."""
import os

from absl import app
from absl import flags
from absl import logging
from xjax import xeval
from xjax import xmod
from xjax import xnn
from xjax import xrand
from xjax import xopt

from data import Data
from module import Encoder, Decoder, Discriminator
from module import FeatureInjector, InputInjector, FeatureRandom, InputRandom
from module import AELoss, GenLoss, DiscLoss, DiscLossSigmoid
from model import ATNNFAE, ATNIAE
from optimizer import Momentum
from evaluator import Evaluator
from learner import Learner
from trainer import Trainer


FLAGS = flags.FLAGS

flags.DEFINE_string('data_file_train', 'data/obama/train.h5',
                    'Train data file.')
flags.DEFINE_string('data_file_valid', 'data/obama/valid.h5',
                    'Valid data file.')
flags.DEFINE_integer('data_batch', 16, 'Data batch size.')
flags.DEFINE_integer('data_step', 16, 'Data step size.')
flags.DEFINE_integer('data_min', 1, 'Data minimum length.')
flags.DEFINE_integer('data_max', 256, 'Data maximum length.')
flags.DEFINE_boolean('data_cache', True, 'Data cache.')

flags.DEFINE_integer('enc_level', 4,'Encoder pooling levels.')
flags.DEFINE_integer('enc_depth', 1, 'Encoder layers in each level.')
flags.DEFINE_integer('enc_input', 256, 'Encoder input dimension.')
flags.DEFINE_integer('enc_feature', 256, 'Encoder feature dimension.')
flags.DEFINE_integer('enc_output', 256, 'Encoder output dimension.')
flags.DEFINE_list('enc_kernel', '3', 'Encoder kernel size.')
flags.DEFINE_list('enc_pool', '2', 'Encoder pooling size.')
flags.DEFINE_float('enc_sigma', 0.000001, 'Encoder initialization.')
flags.DEFINE_enum('enc_transfer', 'relu', ['relu', 'tanh'],
                  'Encoder transfer function.')

flags.DEFINE_integer('dec_level', 4, 'Decoder pooling levels.')
flags.DEFINE_integer('dec_depth', 1, 'Decoder layers in each level.')
flags.DEFINE_integer('dec_input', 256,'Decoder input dimension.')
flags.DEFINE_integer('dec_feature', 256, 'Decoder feature dimension.')
flags.DEFINE_integer('dec_output', 256, 'Decoder output dimension.')
flags.DEFINE_list('dec_kernel', '3', 'Decoder kernel size.')
flags.DEFINE_list('dec_stride', '2', 'Decoder stride size.')
flags.DEFINE_float('dec_sigma', 0.000001, 'Decoder initialization.')
flags.DEFINE_enum('dec_transfer', 'relu', ['relu', 'tanh'],
                  'Decoder transfer function.')

flags.DEFINE_integer('disc_level', 4, 'Discriminator pooling levels.')
flags.DEFINE_integer('disc_depth', 1, 'Discriminator layers in each level.')
flags.DEFINE_integer('disc_input', 256, 'Discriminator input dimension.')
flags.DEFINE_integer('disc_feature', 256, 'Discriminator feature dimension.')
flags.DEFINE_integer('disc_output', 256, 'Discriminator output dimension.')
flags.DEFINE_list('disc_kernel', '3', 'Discriminator kernel size.')
flags.DEFINE_list('disc_pool', '2', 'Discriminator pooling size.')
flags.DEFINE_float('disc_dropout', 0.5, 'Discriminator dropout probability.')
flags.DEFINE_float('disc_sigma', 0.000001, 'Discriminator initialization.')
flags.DEFINE_enum('disc_transfer', 'relu', ['relu', 'tanh'],
                  'Discriminator transfer function.')

flags.DEFINE_float('inj_beta', 0.1, 'Injector noise random level.')

flags.DEFINE_float('ae_loss_weight', 1, 'Autoencoder loss weight.')

flags.DEFINE_float('gen_loss_weight', 1, 'Generator loss weight.')

flags.DEFINE_float('disc_loss_weight', 1, 'Discriminator loss weight.')

flags.DEFINE_float('opt_rate', 0.01, 'Autoencoder learning rate.')
flags.DEFINE_float('opt_coeff', 0.9, 'Autoencoder momentum coefficient.')
flags.DEFINE_float('opt_decay', 0.00001, 'Autoencoder weight decay.')

# Each epoch is a number of training steps and testing steps that randomly
# sample data. This definition is suitable if the dataset is too large and
# iterating over all of it is infeasible.
flags.DEFINE_integer('trainer_train_steps', 100000, 'Train steps per epoch.')
flags.DEFINE_integer('trainer_test_steps', 10000,  'Test steps per epoch.')
flags.DEFINE_integer('trainer_epochs', 1000, 'Number of epoches to run.')
flags.DEFINE_integer('trainer_interval', 10, 'Interval for printing updates.')

flags.DEFINE_string('main_checkpoint', 'checkpoint/obama',
                    'Checkpoint location.')
flags.DEFINE_enum('main_disc_loss', 'logcosh', ['logcosh', 'sigmoid'],
                  'The type of discriminator loss.')
flags.DEFINE_enum('main_model', 'atnnfae', ['atnnfae', 'atniae'],
                  'The type of model.')


def get_transfer(name):
    if name == 'relu':
        return xnn.ReLU
    elif name == 'tanh':
        return xnn.Tanh


def main(unused_argv):
    logging.get_absl_handler().setFormatter(None)
    logging.info('Load train data from %s', FLAGS.data_file_train)
    data_train = Data(FLAGS.data_file_train, FLAGS.data_batch, FLAGS.data_step,
                      FLAGS.data_min, FLAGS.data_max, FLAGS.data_cache)
    logging.info('Load valid data from %s', FLAGS.data_file_valid)
    data_valid = Data(FLAGS.data_file_valid, FLAGS.data_batch, FLAGS.data_step,
                      FLAGS.data_min, FLAGS.data_max, FLAGS.data_cache)
    enc_kernel = tuple(int(k) for k in FLAGS.enc_kernel)
    enc_pool = tuple(int(p) for p in FLAGS.enc_pool)
    encoder = Encoder(FLAGS.enc_level, FLAGS.enc_depth, FLAGS.enc_input,
                      FLAGS.enc_feature, FLAGS.enc_output, enc_kernel, enc_pool,
                      FLAGS.enc_sigma, get_transfer(FLAGS.enc_transfer))
    dec_kernel = tuple(int(k) for k in FLAGS.dec_kernel)
    dec_stride = tuple(int(s) for s in FLAGS.dec_stride)
    decoder = Decoder(FLAGS.dec_level, FLAGS.dec_depth, FLAGS.dec_input,
                      FLAGS.dec_feature, FLAGS.dec_output, dec_kernel,
                      dec_stride, FLAGS.dec_sigma,
                      get_transfer(FLAGS.dec_transfer))
    disc_kernel = tuple(int(k) for k in FLAGS.disc_kernel)
    disc_pool = tuple(int(p) for p in FLAGS.disc_pool)
    discriminator = Discriminator(
        FLAGS.disc_level, FLAGS.disc_depth, FLAGS.disc_input,
        FLAGS.disc_feature, FLAGS.disc_output, disc_kernel, disc_pool,
        FLAGS.disc_dropout, FLAGS.disc_sigma, get_transfer(FLAGS.disc_transfer))
    ae_loss = AELoss(FLAGS.ae_loss_weight)
    gen_loss = GenLoss(FLAGS.gen_loss_weight)
    if FLAGS.main_disc_loss == 'sigmoid':
        disc_loss = DiscLossSigmoid(FLAGS.disc_loss_weight)
    elif FLAGS.main_disc_loss == 'logcosh':
        disc_loss = DiscLoss(FLAGS.disc_loss_weight)
    if FLAGS.main_model == 'atnnfae':
        injector = FeatureInjector(FLAGS.inj_beta)
        random = FeatureRandom()
        model = xmod.jit(xmod.vmap(ATNNFAE(
            encoder, decoder, discriminator, injector, random, ae_loss,
            gen_loss, disc_loss), FLAGS.data_batch))
        enc_opt = Momentum(encoder.params, FLAGS.opt_rate, FLAGS.opt_coeff,
                           FLAGS.opt_decay)
        dec_opt = Momentum(decoder.params, FLAGS.opt_rate, FLAGS.opt_coeff,
                           FLAGS.opt_decay)
        disc_opt = Momentum(discriminator.params, FLAGS.opt_rate,
                            FLAGS.opt_coeff, FLAGS.opt_decay)
        optimizer = xopt.jit(xopt.vmap(xopt.Container(enc_opt, dec_opt, disc_opt)))
    elif FLAGS.main_model == 'atniae':
        injector = InputInjector(FLAGS.enc_input, FLAGS.inj_beta)
        random = InputRandom(FLAGS.enc_input)
        autoencoder = xnn.Sequential(encoder, decoder)
        model = xmod.jit(xmod.vmap(ATNIAE(
            autoencoder, discriminator, injector, random, ae_loss, gen_loss,
            disc_loss), FLAGS.data_batch))
        autoencoder_opt = Momentum(autoencoder.params, FLAGS.opt_rate,
                                   FLAGS.opt_coeff, FLAGS.opt_decay)
        disc_opt = Momentum(discriminator.params, FLAGS.opt_rate,
                            FLAGS.opt_coeff, FLAGS.opt_decay)
        optimizer = xopt.jit(xopt.vmap(xopt.Container(autoencoder_opt, disc_opt)))
    evaluator = xeval.jit(xeval.vmap(Evaluator(), FLAGS.data_batch))
    learner = Learner(optimizer, model, None, evaluator)
    checkpoint = os.path.join(
        FLAGS.main_checkpoint,
        '{}_resconv-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            FLAGS.main_model, FLAGS.enc_level, FLAGS.enc_depth, FLAGS.enc_input,
            FLAGS.enc_feature, FLAGS.enc_output, '-'.join(FLAGS.enc_kernel),
            '-'.join(FLAGS.enc_pool), FLAGS.enc_sigma, FLAGS.enc_transfer)
        + '_resdeconv-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            FLAGS.dec_level, FLAGS.dec_depth, FLAGS.dec_input,
            FLAGS.dec_feature, FLAGS.dec_output, '-'.join(FLAGS.dec_kernel),
            '-'.join(FLAGS.dec_stride), FLAGS.dec_sigma, FLAGS.dec_transfer)
        + '_dense-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            FLAGS.disc_level, FLAGS.disc_depth, FLAGS.disc_input,
            FLAGS.disc_feature, FLAGS.disc_output, '-'.join(FLAGS.disc_kernel),
            '-'.join(FLAGS.disc_pool), FLAGS.disc_dropout, FLAGS.disc_sigma,
            FLAGS.disc_transfer)
        + '_feat-{}'.format(FLAGS.inj_beta)
        + '_feat_plusmax-{}'.format(FLAGS.ae_loss_weight)
        + '_logcosh-{}'.format(FLAGS.gen_loss_weight)
        + '_{}-{}'.format(FLAGS.main_disc_loss, FLAGS.disc_loss_weight)
        + '_mom-{}-{}-{}'.format(
            FLAGS.opt_rate, FLAGS.opt_coeff, FLAGS.opt_decay)
        + '_byte-{}-{}-{}-{}'.format(
            FLAGS.data_batch, FLAGS.data_step, FLAGS.data_min, FLAGS.data_max))
    run = Trainer(learner, data_train, data_valid, FLAGS.trainer_train_steps,
                  FLAGS.trainer_test_steps, FLAGS.trainer_epochs,
                  FLAGS.trainer_interval, checkpoint)
    run()


if __name__ == '__main__':
    app.run(main)
