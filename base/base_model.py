import tensorflow as tf
from utils.logger import get_logger

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.log = get_logger()
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess, checkpoint_dir=None, global_step=None):
        if(checkpoint_dir is None):
            checkpoint_dir_ = self.config.checkpoint_dir
        else:
            checkpoint_dir_ = checkpoint_dir
        if(global_step is None):
            global_step_ = self.global_step
        else:
            global_step_ = global_step

        self.log.info("Saving model...")
        saved = self.saver.save(sess, checkpoint_dir_, global_step=global_step_)
        self.log.info("Model saved")
        return saved

    def load(self, sess, checkpoint_dir=None):
        if(checkpoint_dir is None):
            latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        else:
            latest_checkpoint = checkpoint_dir

        if latest_checkpoint:
            self.log.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            self.log.info("Model loaded")
        else:
            self.log.info("No checkpoint found")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def add_savers(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError   