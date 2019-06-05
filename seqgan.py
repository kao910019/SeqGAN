# -*- coding: utf-8 -*-
import os
import sys
import time

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import control_flow_ops

from hparams import HParams
from discriminator import Discriminator
from generator import Generator
from tokenized_data import TokenizedData
    
#Load settings file
sys.path.append("..")
from settings import SYSTEM_ROOT
from settings import RESULT_DIR, RESULT_FILE, TRAIN_LOG_DIR, TEST_LOG_DIR, INFER_LOG_DIR
#from settings import EMOTION_TYPES, EMOTION_LENGTH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_sequence = ['pre-train','dis-train','gan-train']

np.set_printoptions(threshold=np.inf)

class SeqGAN(object):
    
    def __init__(self, session, training, hparams = None):
        self.session = session
        self.training = training
        print("# Prepare dataset placeholder and hyper parameters ...")
        #Load Hyper-parameters file
        if hparams is None:
            self.hparams = HParams(SYSTEM_ROOT).hparams
        else:
            self.hparams = hparams
        # Initializer
        initializer = self.get_initializer(self.hparams.init_op, 
                                           self.hparams.random_seed, 
                                           self.hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        
        self.tokenized_data = TokenizedData(hparams = self.hparams, training = self.training)
        self.vocab_list = self.tokenized_data.vocab_list
        self.vocab_size = self.tokenized_data.vocab_size
        self.vocab_table = self.tokenized_data.vocab_table
        self.reverse_vocab_table = self.tokenized_data.reverse_vocab_table
        
        if self.training:
            self.build_train_model()
            #tensorboard
            self.train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR+self.train_mode, self.session.graph)
            self.test_summary_writer = tf.summary.FileWriter(TEST_LOG_DIR+self.train_mode)
        else:
            self.build_predict_model()
            #Tensorboard
            tf.summary.FileWriter(INFER_LOG_DIR, self.session.graph)
            
    def get_initializer(self, init_op, seed=None, init_weight=None):
        """Create an initializer. init_weight is only for uniform."""
        if init_op == "uniform":
            assert init_weight
            return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
        elif init_op == "glorot_normal":
            return tf.contrib.keras.initializers.glorot_normal(seed=seed)
        elif init_op == "glorot_uniform":
            return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
        else:
            raise ValueError("Unknown init_op %s" % init_op)
# =============================================================================
# Create model
# =============================================================================
                
    def build_model(self, batch_input, mode = 'inference'):
        print("# Build model =",mode)
        with tf.variable_scope('Emotion_SeqGAN'):
            
            with tf.variable_scope("embeddings", dtype = tf.float32):
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.hparams.embedding_size], tf.float32)
            
            g_model = Generator(mode = mode,
                                    hparams = self.hparams,
                                    tokenized_data = self.tokenized_data,
                                    embedding = self.embedding,
                                    batch_input = batch_input)
            d_model = Discriminator(mode = mode,
                                    hparams = self.hparams,
                                    tokenized_data = self.tokenized_data,
                                    embedding = self.embedding,
                                    batch_input = batch_input,
                                    generator = g_model)
            
            global_step = tf.math.add((g_model.pre_train_global_step + g_model.gan_train_global_step),
                                      d_model.dis_train_global_step, name = 'total_global_step')
            
            if mode != 'inference':
                g_model.gan_connect(discriminator = d_model, mode = mode)
                
        return g_model, d_model, global_step
    
    def get_tensors_in_checkpoint_file(self, file_name, all_tensors=True, tensor_name=None):
        varlist, var_value = [], []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
          var_to_shape_map = reader.get_variable_to_shape_map()
          for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
        else:
            varlist.append(tensor_name)
            var_value.append(reader.get_tensor(tensor_name))
        return (varlist, var_value)

    def build_tensors_in_checkpoint_file(self, loaded_tensors):
        full_var_list = []
        for i, tensor_name in enumerate(loaded_tensors[0]):
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
                full_var_list.append(tensor_aux)
            except:
                print('* Not found: '+tensor_name)
        return full_var_list
    
    def variable_loader(self):
        self.ckpt = tf.train.get_checkpoint_state(RESULT_DIR)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            print("# Find checkpoint file:", self.ckpt.model_checkpoint_path)
            restored_vars  = self.get_tensors_in_checkpoint_file(file_name = self.ckpt.model_checkpoint_path)
            tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
            self.saver = tf.train.Saver(tensors_to_load, max_to_keep=4, keep_checkpoint_every_n_hours=1.0)
            if self.training == True:
                if input("# Keep training? [y/n]: ") not in ["no","n"]:
                    print("# Restoring model weights ...")
                    self.saver.restore(self.session, self.ckpt.model_checkpoint_path)
                    return True
            else:
                self.saver.restore(self.session, self.ckpt.model_checkpoint_path)
                return True
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4, keep_checkpoint_every_n_hours=1.0)
        return False
        
    def build_train_model(self):
        
        with self.session.graph.as_default():
            
            train_mode = input("# Select train mode [pre/dis/gan]: ")
            if train_mode in ['pre-train','pre','p']:
                self.train_mode = 'pre-train'
            elif train_mode in ['dis-train','dis','d']:
                self.train_mode = 'dis-train'
            elif train_mode in ['gan-train','gan','g']:
                self.train_mode = 'gan-train'
                
            with tf.variable_scope('Network_Operator'):
                self.loss_log_var = tf.Variable(0.0, name = "loss_log_var", trainable=False)
                tf.summary.scalar(self.train_mode+"_loss", self.loss_log_var)
                self.write_loss_op = tf.summary.merge_all()
                self.dataset_handler = tf.placeholder(tf.string, shape=[], name='dataset_handler')
                self.train_batch_iter = self.tokenized_data.get_training_batch(self.tokenized_data.train_id_set)
                self.test_batch_iter = self.tokenized_data.get_training_batch(self.tokenized_data.test_id_set)
                input_batch = self.tokenized_data.multiple_batch(self.dataset_handler, self.train_batch_iter.batched_dataset)
                
            self.g_model, self.d_model, self.global_step = self.build_model(batch_input = input_batch, mode = self.train_mode)
            
            if self.train_mode == 'pre-train':
                self.epoch_num = self.g_model.pre_train_epoch
            elif self.train_mode == 'dis-train':
                self.epoch_num = self.d_model.dis_train_epoch
            elif self.train_mode == 'gan-train':
                self.epoch_num = self.g_model.gan_train_epoch
        
            self.restore = self.variable_loader()
            
    def build_predict_model(self):
        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name = 'NN_input')
        src_dataset = tf.data.Dataset.from_tensor_slices(self.src_placeholder)
        self.infer_batch = self.tokenized_data.get_inference_batch(src_dataset)
        print("# Creating inference model ...")
        self.g_model, self.d_model, _ = self.build_model(self.infer_batch, mode = 'inference')
        print("# Restoring model weights ...")
        self.restore = self.variable_loader()
        assert self.restore
        self.session.run(tf.tables_initializer())
# =============================================================================
# Create model
# =============================================================================
    def train(self):
        
        hp_epoch_times = self.hparams.num_epochs
        train_step, first_step = True, True
        
        print("# Get dataset handler.")
        training_handle = self.session.run(self.train_batch_iter.handle)
        testing_handle = self.session.run(self.test_batch_iter.handle)
        training_handle = {self.dataset_handler: training_handle}
        testing_handle = {self.dataset_handler: testing_handle}
        self.session.run([self.train_batch_iter.initializer, self.test_batch_iter.initializer])
        self.session.run(tf.tables_initializer())
        
        if not self.restore:
            self.session.run(tf.global_variables_initializer(), feed_dict = training_handle)
            print("# Load embedding from Word2Vec model.")
                  
        # Initialize the statistic variables
        train_perp, last_record_perp = 2000.0, 2.0
        train_epoch_times = 0
            
        global_step = self.global_step.eval(session=self.session)
        epoch_num = self.epoch_num.eval(session=self.session)
        print("# Global step =", global_step)
        print("="*50)
        print("# Training loop started @ {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        print("# Epoch training {} times.".format(hp_epoch_times))
        
        while train_epoch_times < hp_epoch_times:
            # Train step
            epoch_start_time = time.time()
            ckpt_loss, ckpt_count  = 0.0, 0.0
            learning_rate = self._get_learning_rate(train_perp)
            
            if train_step:
                print("# Start training step.")
                self.session.run(self.train_batch_iter.initializer)
                dataset_handler = training_handle
            else:
                print("# Start testing step.")
                self.session.run(self.test_batch_iter.initializer)
                dataset_handler = testing_handle
                
            while True:
                try:
                    
                    if train_step:
                        step_loss, batch_size, _, global_step = self.update(learning_rate, dataset_handler)
                    else:
                        step_loss, batch_size = self.test(dataset_handler)
                        
                    if train_step and ((global_step % 100 == 0) or first_step):
                        print("Step:", global_step,"loss =", step_loss)
                        first_step = False
                        
                    ckpt_loss += (step_loss * batch_size)
                    ckpt_count += batch_size
                    
                except tf.errors.OutOfRangeError:
                    epoch_dur = time.time() - epoch_start_time
                    epoch_loss = ckpt_loss / ckpt_count
                    
                    if train_step:
                        print("# Train epoch:", epoch_num,"loss =", epoch_loss)
                        self.write_summary(self.session, self.train_summary_writer, epoch_loss, epoch_num)
                        print("# Train step {:5d} @ {} | {:.2f} seconds elapsed."
                          .format(global_step, time.strftime("%Y-%m-%d %H:%M:%S"), round(epoch_dur, 2)))
                    else:
                        print("# Test epoch:", epoch_num,"loss =", epoch_loss)
                        train_epoch_times += 1
                        self.write_summary(self.session, self.test_summary_writer, epoch_loss, epoch_num)
                        epoch_num = self.session.run(tf.assign_add(self.epoch_num, 1))
                        print("# Finished epoch {:2d}/{:2d} @ {} | {:.2f} seconds elapsed."
                          .format(train_epoch_times, hp_epoch_times, time.strftime("%Y-%m-%d %H:%M:%S"), round(epoch_dur, 2)))
                        self.saver.save(self.session, RESULT_FILE, global_step = global_step)
                        
                    if train_perp < last_record_perp:
                        last_record_perp = train_perp
                    break
            # Turn to test step
            train_step = not train_step
        self.train_summary_writer.close()
        self.test_summary_writer.close()
    
    def update(self, learning_rate, dataset_handler):
        if self.train_mode =='pre-train':
            result = self.g_model.pre_train_update(self.session, learning_rate, dataset_handler)
        elif self.train_mode == 'dis-train':
            result = self.d_model.dis_train_update(self.session, learning_rate, dataset_handler)
        elif self.train_mode == 'gan-train':
            result = self.g_model.gan_train_update(self.session, learning_rate, dataset_handler)
            
#            self.g_model.test(self.session, learning_rate, dataset_handler, self.global_step)
            
        global_step = self.session.run(self.global_step)
        step_loss, batch_size, step_summary, local_step = result
        self.train_summary_writer.add_summary(step_summary, global_step)
        return step_loss, batch_size, local_step, global_step
    
    def test(self, dataset_handler):
        if self.train_mode =='pre-train':
            result = self.g_model.pre_train_test(self.session, dataset_handler)
        elif self.train_mode == 'dis-train':
            result = self.d_model.dis_train_test(self.session, dataset_handler)
        elif self.train_mode == 'gan-train':
            result = self.g_model.gan_train_test(self.session, dataset_handler)
        step_loss, batch_size = result
        return step_loss, batch_size
        
    def write_summary(self, session, summary_writer, epoch_loss, epoch):
        summary = session.run(self.write_loss_op, {self.loss_log_var: epoch_loss})
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
            
    @staticmethod
    def _get_learning_rate(perplexity):
#        if perplexity <= 1.48:
#            return 9.6e-5
#        elif perplexity <= 1.64:
#            return 1e-4
#        elif perplexity <= 2.0:
#            return 1.2e-4
#        elif perplexity <= 2.4:
#            return 1.6e-4
#        elif perplexity <= 3.2:
#            return 2e-4
#        elif perplexity <= 4.8:
#            return 2.4e-4
#        elif perplexity <= 8.0:
#            return 3.2e-4
#        elif perplexity <= 16.0:
#            return 4e-4
#        elif perplexity <= 32.0:
#            return 6e-4
#        else:
        return 8e-4
    
    def generate(self, question):
        
        tokens = nltk.word_tokenize(question.lower())
        sentence = [' '.join(tokens[:]).strip()]
        
        outputs = self.g_model.generate(self.session, feed_dict={self.src_placeholder: sentence})
        
        if self.hparams.beam_width > 0:
            outputs = outputs[0]
        eos_token = self.hparams.eos_token.encode("utf-8")
        outputs = outputs.tolist()[0]
        if eos_token in outputs:
            outputs = outputs[:outputs.index(eos_token)]
        outputs = b' '.join(outputs).decode('utf-8')
        
        return outputs

if __name__ == "__main__":

    with tf.Session() as sess:
        print("# Start")
        model = SeqGAN(sess, training = False)
        print("# Generate")
        while True:
            sentence = input("Q: ")
            answer = model.generate(sentence)
            print('-'*20)
            print("A:", answer)
            print('-'*20)
    
    