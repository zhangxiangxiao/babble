"""Unit test for trainer."""

from trainer import Trainer

from absl.testing import absltest
from xjax import xrand
from xjax import xeval
from xjax import xopt
from xjax import xmod

from data import Data
from module import Encoder, Decoder, Discriminator
from module import FeatureInjector, FeatureRandom
from module import AELoss, GenLoss, DiscLoss
from model import ATNNFAE
from optimizer import Momentum
from evaluator import Evaluator
from learner import Learner


class TrainerTest(absltest.TestCase):
    def setUp(self):
        data_train = Data('data/obama/train.h5', 4, 16, 1, 64, True)
        data_valid = Data('data/obama/valid.h5', 4, 16, 1, 64, True)
        encoder = Encoder(2, 2, 256, 256, 256)
        decoder = Decoder(2, 2, 256, 256, 256)
        discriminator = Discriminator(2, 2, 256, 256, 256)
        injector = FeatureInjector(1)
        random = FeatureRandom()
        ae_loss = AELoss(1)
        gen_loss = GenLoss(0.1)
        disc_loss = DiscLoss(1)
        model = xmod.jit(xmod.vmap(ATNNFAE(
            encoder, decoder, discriminator, injector, random, ae_loss,
            gen_loss, disc_loss), 4))
        enc_opt = Momentum(encoder.params, 0.01, 0.9, 0.00001)
        dec_opt = Momentum(decoder.params, 0.01, 0.9, 0.00001)
        disc_opt = Momentum(discriminator.params, 0.01, 0.9, 0.1)
        optimizer = xopt.jit(xopt.vmap(xopt.Container(
            enc_opt, dec_opt, disc_opt)))
        evaluator = xeval.jit(xeval.vmap(Evaluator(), 4))
        learner = Learner(optimizer, model, None, evaluator)
        self.run = Trainer(learner, data_train, data_valid, 10, 2, 2, 0,
                           'checkpoint/unittest')

    def test_run(self):
        run = self.run
        run()


if __name__ == '__main__':
    absltest.main()
