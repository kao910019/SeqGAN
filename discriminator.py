# -*- coding: utf-8 -*-
import sys

import tensorflow as tf
import numpy as np

class Discriminator(object):
    
    def __init__(self, mode, hparams, tokenized_data, embedding, batch_input, generator):
        
        self.mode = mode
        self.hparams = hparams
        
        self.sample_times = self.hparams.sample_times if self.hparams.reward_type == 'MC_Search' else 1
#        self.sample_times = self.hparams.sample_times if self.mode == 'gan-train' else 1
        if mode == 'inference':
            self.training = False
        else:
            self.training = True
        
        self.embedding = embedding
        #Word2Vec embedding model
        self.embedding_model = tokenized_data.embedding_model
        
        self.vocab_list = tokenized_data.vocab_list
        self.vocab_size = tokenized_data.vocab_size
        self.vocab_table = tokenized_data.vocab_table
        self.reverse_vocab_table = tokenized_data.reverse_vocab_table
        
        self.batch_input = batch_input
        self.time_major = self.hparams.time_major
        
        self.generator = generator
        
        self.batch_size = tf.size(self.batch_input.source_sequence_length)
        
        with tf.variable_scope('Discriminator/placeholder'):
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.dis_train_epoch = tf.Variable(1, name = 'dis_train_epoch', trainable=False)
            self.dis_train_global_step = tf.Variable(0, name = 'dis_train_global_step', trainable=False)
                
        self.labels, self.logits = self.build_model(self.hparams)
        with tf.name_scope('Reward'):
            self.poss = tf.nn.softmax(self.logits)[:, 1]
#            self.reward = tf.concat([tf.slice(self.poss, [0], [35]), -tf.slice(self.poss, [35], [35])], axis = 0)
                
        if self.training == True:
            self.train(self.hparams, self.labels, self.logits)
        else:
            pass
    
# =============================================================================
# Training model
# =============================================================================
    def train(self, hparams, labels, logits):
        """To train discriminator model."""
        with tf.name_scope("D_loss"):
            self.dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            result = tf.argmax(logits, axis=1)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(result, labels), tf.float32))
            
        with tf.variable_scope("D_Gradient"):
            dis_gradients = tf.gradients(self.dis_loss, self.dis_params)
            dis_grad, self.dis_gradient_norm_summary = self.gradient_clip(dis_gradients, hparams.max_gradient_norm, add_string = "dis_")
            dis_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.dis_update = dis_opt.apply_gradients(
                    zip(dis_gradients, self.dis_params), global_step = self.dis_train_global_step)
            
            print("# Trainable Discriminator variables:")
            for dis_param in self.dis_params:
                print("  {}, {}, {}".format(dis_param.name, str(dis_param.get_shape()), dis_param.op.device))
            
            # Tensorboard
            self.dis_train_summary = tf.summary.merge([
                    tf.summary.scalar("dis_learning_rate", self.learning_rate),
                    tf.summary.scalar("dis_loss", self.dis_loss),
                    tf.summary.scalar("accuracy", self.acc),] \
                    + self.dis_gradient_norm_summary)
            
    def gradient_clip(self, gradients, max_gradient_norm, add_string = ""):
        """Clipping gradients of model."""
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        gradient_norm_summary = [tf.summary.scalar(add_string+"grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar(add_string+"clipped_gradient", tf.global_norm(clipped_gradients)))
        return clipped_gradients, gradient_norm_summary
    
    def dis_train_update(self, sess, learning_rate, dataset_handler):
        """update model params"""
        assert self.training
        feed_dict={self.learning_rate: learning_rate}
        feed_dict.update(dataset_handler)
        
        loss, batch_size, summary, global_step, _ = sess.run([self.dis_loss, self.batch_size, 
                                                              self.dis_train_summary, self.dis_train_global_step,
                                                              self.dis_update],
                                                              feed_dict=feed_dict)
        return loss, batch_size, summary, global_step
    
    def dis_train_test(self, sess, dataset_handler):
        """update model params"""
        assert self.training
        loss, batch_size = sess.run([self.dis_loss, self.batch_size], feed_dict = dataset_handler)
        return loss, batch_size
    
    def test(self, sess, learning_rate, dataset_handler):
        """pretrain model params"""
        assert self.training
        feed_dict={self.learning_rate: learning_rate}
        feed_dict.update(dataset_handler)
        
        test1, test2, test3, test4, test5 = sess.run([self.question, self.response, self.ques_length, self.resp_length, self.labels],
                feed_dict=feed_dict)
        
        print(np.shape(test1))
        print(np.shape(test2))
        print(np.shape(test3))
        print(np.shape(test4))
        print(np.shape(test5))
        
        sys.exit()
        
# =============================================================================
# Create model
# =============================================================================
    def build_model(self, hparams):
        
        with tf.variable_scope('Discriminator') as scope:
            
            self.question, self.response, self.ques_length, self.resp_length, self.labels = self.build_encoder_input(hparams)
            
            ques_output, ques_state = self.build_encoder(hparams, source = self.question, length = self.ques_length, string = 'question')
            resp_output, resp_state = self.build_encoder(hparams, source = self.response, length = self.resp_length, string = 'response')
            
            if hparams.num_layers > 1:
                ques_state_concat = tf.reshape([state for state in ques_state],[hparams.num_layers, -1, hparams.num_units])
                resp_state_concat = tf.reshape([state for state in resp_state],[hparams.num_layers, -1, hparams.num_units])
                state_concat = tf.concat([ques_state_concat, resp_state_concat], axis = 0)
            else:
                state_concat = tf.reshape([ques_state, resp_state],[2, -1, hparams.num_units])
            final_output, final_state = self.build_sentence_encoder(hparams, source = state_concat)
            if hparams.num_layers > 1:
                final_state = tf.concat([state for state in final_state], axis=1)
            logits = tf.layers.dense(final_state, 2)
            
            self.dis_params = scope.trainable_variables()
            
        return self.labels, logits
    
    def build_encoder_input(self, hparams):
        
        with tf.variable_scope('encoder_input'):
            
            if self.mode == 'dis-train':
                if self.hparams.reward_type == 'Partial_Reward':
                    # Reward partial sentence.
                    src_question = self.generator.source_sample
                    tgt_response = self.generator.partial_target_id
                    src_length = self.generator.source_sample_sequence_length
                    gen_response = self.generator.partial_sample_id
                else:
                    # Reward sample sentence.
                    src_question = self.batch_input.source
                    tgt_response = self.batch_input.target_output
                    src_length = self.batch_input.source_sequence_length
                    tgt_length = self.batch_input.target_sequence_length
                    gen_response = self.generator.result_sample_id
                    
                gen_length = tf.cast(tf.argmin(gen_response, 0), tf.int32)
                    
                if self.time_major:
                    src_question = tf.transpose(src_question)
                    tgt_response = tf.transpose(tgt_response)
                if self.hparams.reward_type == 'Partial_Reward':
                    tgt_length = tf.cast(tf.argmin(tgt_response, 0), tf.int32)
                    gen_response = tf.slice(gen_response, [0, 0], [tf.add(tf.shape(tgt_response)[0],-1), -1])
                    tgt_response = tf.slice(tgt_response, [0, 0], [tf.add(tf.shape(tgt_response)[0],-1), -1])
                    
                #batch_size * 2
                question = tf.concat([src_question, src_question], axis=1, name = "question_batch")
                ques_length = tf.concat([src_length, src_length], axis=0, name = "ques_length")
                response = tf.concat([tgt_response, gen_response], axis=1, name = "response_batch")
                resp_length = tf.concat([tgt_length, gen_length], axis=0, name = "resp_length")
                labels = tf.concat([tf.ones(tf.size(tgt_length), dtype=tf.int64),
                                    tf.zeros(tf.size(gen_length), dtype=tf.int64)], axis=0, name = "label_batch")
            
            
                return question, response, ques_length, resp_length, labels
            else:
                if self.training == True:
                    # Reward partial sentence.
                    src_question = self.generator.source_sample
                    src_length = self.generator.source_sample_sequence_length
#                    gen_response = self.generator.sample_id
                    gen_response = self.generator.partial_sample_id
                    labels = tf.concat([tf.ones(tf.size(src_length), dtype=tf.int64),
                                        tf.zeros(tf.size(src_length), dtype=tf.int64)], axis=0, name = "label_batch")
            
                else:
                    # inference mode.
                    src_question = self.batch_input.source
                    src_length = self.batch_input.source_sequence_length
                    gen_response = self.generator.sample_id[:,:,0]
                    labels = None
                    
                gen_length = tf.cast(tf.argmin(gen_response, 0), tf.int32)
                
                if self.time_major:
                    src_question = tf.transpose(src_question)
                    
                return src_question, gen_response, src_length, gen_length, labels
    
    def build_encoder(self, hparams, source, length, string):
        """Create encoder."""
            
        with tf.variable_scope('Encoder_'+string) as scope:
            
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, source)
            
            # Encoder_outpus: [max_time, batch_size, num_units]
            cell = self.create_rnn_cell(hparams)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,#CELL
                    encoder_emb_inp,#INPUTS
                    dtype=scope.dtype,
                    sequence_length= length,
                    time_major=self.time_major,
                    scope = scope)
                
        return encoder_outputs, encoder_state
    
    def build_sentence_encoder(self, hparams, source):
        """Create encoder."""
            
        with tf.variable_scope('Encoder_sentence') as scope:
            
            # Encoder_outpus: [max_time, batch_size, num_units]
            cell = self.create_rnn_cell(hparams)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,#CELL
                    source,#INPUTS
                    dtype=scope.dtype,
#                    sequence_length= length,
                    sequence_length= None,
                    time_major=self.time_major,
                    scope = scope)
                
        return encoder_outputs, encoder_state
    
    
    def create_rnn_cell(self, hparams):
        """Create multi-layer RNN cell."""
        cell_list = []
        for i in range(hparams.num_layers):
            #Create a single RNN cell
            if hparams.rnn_cell_type == 'LSTM':
                single_cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units, state_is_tuple=True)
            else:
                single_cell = tf.contrib.rnn.GRUCell(hparams.num_units)
                
            if hparams.keep_prob < 1.0:
                single_cell = tf.contrib.rnn.DropoutWrapper(cell = single_cell, 
                                                            input_keep_prob = hparams.keep_prob)
            cell_list.append(single_cell)
            
        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)