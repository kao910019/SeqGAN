# -*- coding: utf-8 -*-
import sys

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class Generator(object):
    """Generator model, base on Seq2Seq."""
    def __init__(self, mode, hparams, tokenized_data, embedding, batch_input):
        
        self.mode = mode
        if mode == 'inference':
            self.training = False
        else:
            self.training = True
            
        self.hparams = hparams
        self.embedding = embedding
        #Word2Vec embedding model
        self.embedding_model = tokenized_data.embedding_model
        
        self.vocab_list = tokenized_data.vocab_list
        self.vocab_size = tokenized_data.vocab_size
        self.vocab_table = tokenized_data.vocab_table
        self.reverse_vocab_table = tokenized_data.reverse_vocab_table
        
        self.batch_input = batch_input
        self.batch_size = tf.size(self.batch_input.source_sequence_length)
        self.sample_times = self.hparams.sample_times if self.hparams.reward_type == 'MC_Search' else 1
        self.time_major = self.hparams.time_major
        
        with tf.variable_scope('Generator/placeholder'):
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            self.pre_train_epoch = tf.Variable(1, name = 'pre_train_epoch', trainable=False)
            self.gan_train_epoch = tf.Variable(1, name = 'gan_train_epoch', trainable=False)
            self.pre_train_global_step = tf.Variable(0, name = 'pre_train_global_step', trainable=False)
            self.gan_train_global_step = tf.Variable(0, name = 'gan_train_global_step', trainable=False)
            
        # Training or inference graph
        print("# Building graph for the model ...")
        self.logits, self.sample_id, self.final_context_state = self.build_model(self.hparams)
        
        if self.mode == 'inference':
            # Generate response to user
            self.sample_words = self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id), name = 'NN_output')
        
#        self.test1 = self.reverse_vocab_table.lookup(tf.to_int64(self.target_id))
#        self.test2 = self.reverse_vocab_table.lookup(tf.to_int64(self.partial_target_id))
        self.test2 = self.reverse_vocab_table.lookup(tf.to_int64(self.result_sample_id))
        self.test3 = self.reverse_vocab_table.lookup(tf.to_int64(self.partial_sample_id1))
        self.test4 = self.reverse_vocab_table.lookup(tf.to_int64(self.partial_sample_id3))
        
    def generate(self, sess, feed_dict):
        
        _, _, result = sess.run([self.batch_input.initializer, self.logits, self.sample_words], feed_dict=feed_dict)
        
        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            result = result.transpose()
            
        return result
    
# =============================================================================
# Training model
# =============================================================================    
    def train(self, hparams, logits, target_output, reward):
        """To train generator model."""
#        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE) as scope:
        with tf.name_scope("G_loss"):
            
            #Get max time
            time_axis = 0 if self.time_major else 1
            self.max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]
            target_weight = tf.sequence_mask(self.batch_input.target_sequence_length, self.max_time, dtype=logits.dtype)
            if self.time_major:
                target_weight = tf.transpose(target_weight)
            self.target_count = tf.reduce_sum(target_weight)
            
            #Pre-train loss
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
            self.pre_loss = tf.reduce_sum(crossent * target_weight) / tf.to_float(self.batch_size)
            self.pre_perplexity = tf.exp(tf.reduce_sum(crossent * target_weight) / self.target_count)
#            self.pre_perplexity = tf.exp(self.pre_loss)
            
            #GAN-train loss
            one_hot = tf.one_hot(target_output, self.vocab_size)
            gan_logits = tf.concat([logits, logits], axis = 1)
            gan_target_weight = tf.concat([target_weight, target_weight], axis = 1)
            self.prob = tf.reduce_sum((one_hot * tf.nn.softmax(gan_logits)), axis = 2)
            self.gan_loss = -tf.reduce_sum(tf.log(tf.math.maximum(self.prob, 1e-20)) * reward * gan_target_weight) / tf.to_float(self.batch_size * 2)
            self.gan_perplexity = tf.exp(self.gan_loss)
            
        with tf.variable_scope("G_Gradient"):
            # Supervised learning
            pre_gradients = tf.gradients(self.pre_loss, self.g_params)
            pre_grad, self.pre_gradient_norm_summary = self.gradient_clip(pre_gradients, hparams.max_gradient_norm, add_string = "pre_")
            pre_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.pre_update = pre_opt.apply_gradients(zip(pre_grad, self.g_params), global_step = self.pre_train_global_step)
            
            # Unsupervised learning
            gan_gradients = tf.gradients(self.gan_loss, self.g_params)
            gan_grad, self.gan_gradient_norm_summary = self.gradient_clip(gan_gradients, hparams.max_gradient_norm, add_string = "gan_")
            gan_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.gan_update = gan_opt.apply_gradients(zip(gan_grad, self.g_params), global_step = self.gan_train_global_step)

            print("# Trainable Generator variables:")
            for g_param in self.g_params:
                print("  {}, {}, {}".format(g_param.name, str(g_param.get_shape()), g_param.op.device))
            
            # Tensorboard
            self.pre_train_summary = tf.summary.merge([
                    tf.summary.scalar("pre_learning_rate", self.learning_rate),
                    tf.summary.scalar("pre_loss", self.pre_loss),
                    tf.summary.scalar("pre_perplexity", self.pre_perplexity)] \
                    + self.pre_gradient_norm_summary)
            self.gan_train_summary = tf.summary.merge([
                    tf.summary.scalar("gan_learning_rate", self.learning_rate),
                    tf.summary.scalar("gan_loss", self.gan_loss),
                    tf.summary.scalar("gan_perplexity", self.gan_perplexity)] \
                    + self.gan_gradient_norm_summary)
            
    def gradient_clip(self, gradients, max_gradient_norm, add_string = ""):
        """Clipping gradients of model."""
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        gradient_norm_summary = [tf.summary.scalar(add_string+"grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar(add_string+"clipped_gradient", tf.global_norm(clipped_gradients)))
        return clipped_gradients, gradient_norm_summary
    
    def pre_train_update(self, sess, learning_rate, dataset_handler):
        """pretrain model params"""
        assert self.training
        feed_dict={self.learning_rate: learning_rate}
        feed_dict.update(dataset_handler)
        
        loss, batch_size, summary, global_step, _ = sess.run([
                self.pre_loss, self.batch_size, self.pre_train_summary, self.pre_train_global_step, self.pre_update],
                feed_dict=feed_dict)
        
        return loss, batch_size, summary, global_step
    
    def test(self, sess, learning_rate, dataset_handler, glo_step):
        """pretrain model params"""
        assert self.training
        feed_dict={self.learning_rate: learning_rate}
        feed_dict.update(dataset_handler)
        loop = 1
        while True:
            try:
            
                bs, mt, test2, test3, test4 = sess.run([self.batch_size, self.ma_time, self.test2, self.test3, self.test4], feed_dict=feed_dict)
                if self.time_major:
                    test3 = test3.transpose()
#                    test4 = test4.transpose()
                    
                
#                print(test2, test2.shape)
#                print("-"*30)
#                print(test3, test3.shape)
#                
#                print("-"*30)
#                print(test4, test4.shape)
                print("loop =",loop)
                print("batch_size =",bs)
                print("max_time =",mt)
                print("test2", test2.shape)
                print("test3", test3.shape)
                print("test4", test4.shape)
                print("="*30)
#                test4 = sess.run(self.test4, feed_dict=feed_dict)
                
#                if self.time_major:
#                    test4 = test4.transpose()
                    
            
            except tf.errors.OutOfRangeError:
                print("OutOfRangeError loop=",loop)
                sys.exit()
            loop += 1
            
            
#        return loss, batch_size, summary, global_step
    
    def pre_train_test(self, sess, dataset_handler):
        assert self.training
        loss, batch_size = sess.run([self.pre_loss, self.batch_size], feed_dict=dataset_handler)
        return loss, batch_size
    
    def gan_train_update(self, sess, learning_rate, dataset_handler):
        """update model params"""
        assert self.training
        feed_dict={self.learning_rate: learning_rate}
        feed_dict.update(dataset_handler)
        
        bs, mt, test3, test4, loss, batch_size, summary, global_step, _ = sess.run([self.batch_size, self.ma_time, self.test3, self.test4,
                self.gan_loss, self.batch_size, self.gan_train_summary, self.gan_train_global_step, self.gan_update],
                feed_dict=feed_dict)
        
        if self.time_major:
            test3 = test3.transpose()
        print("global_step =",global_step)
        print("batch_size =",bs)
        print("max_time =",mt)
        print("test3", test3.shape)
        print("test4", test4.shape)
                
        return loss, batch_size, summary, global_step
    
    def gan_train_test(self, sess, dataset_handler):
        assert self.training
        loss, batch_size = sess.run([self.gan_loss, self.batch_size], feed_dict=dataset_handler)
        return loss, batch_size
    
    def gan_connect(self, discriminator, mode):
        """Connect discriminator to generator."""
        self.discriminator = discriminator
        with tf.name_scope('GAN_connect'):
            # shape = [batch_size, sample_times * time]
            self.reward = tf.transpose(self.discriminator.poss)
            
            if self.mode in ['pre-train','dis-train']:
                target_output = self.batch_input.target_output
            elif self.mode == 'gan-train':
                teacher_forcing = self.batch_input.target_output
                generator_sample = tf.transpose(self.result_sample_id)
                
                time_axis = 1 if self.time_major else 0
                max_time = teacher_forcing.shape[time_axis].value or tf.shape(teacher_forcing)[time_axis]
                
                teacher_forcing_reward = tf.cast(tf.fill([self.batch_size, max_time], 1), tf.float32)
                
                self.reward = tf.reshape(self.reward, [self.batch_size, -1, self.sample_times])
                words_mean_reward = tf.reduce_mean(self.reward, axis = 2, name = 'words_mean_reward')
#                sentence_mean_reward = tf.reduce_mean(words_mean_reward, axis = 1, name = 'sentence_mean_reward')
                
                self.reward = tf.concat([words_mean_reward, teacher_forcing_reward], axis = 0)
                self.reward = tf.subtract(self.reward, 0.5)
                target_output = tf.concat([generator_sample, teacher_forcing], axis = 0)
                
            if self.time_major:
                target_output = tf.transpose(target_output)
            # shape = [times, batch_size]
            self.reward = tf.transpose(self.reward)
            
        self.train(self.hparams, self.logits, target_output, self.reward)
        
# =============================================================================
# Create model
# =============================================================================
    def build_model(self, hparams):
        """Build generator graph."""
        with tf.variable_scope('Generator') as scope:
            #Build Encoder
            encoder_outputs, encoder_state = self.build_encoder(hparams)
            #Build Decoder
            logits, sample_id, final_context_state = self.build_decoder(encoder_outputs, encoder_state, hparams)
            
            self.g_params = scope.trainable_variables()
            
        return logits, sample_id, final_context_state
    
    def build_encoder(self, hparams):
        """Create encoder."""
        
        source = self.batch_input.source
        if self.time_major:
            source = tf.transpose(source)
            
        with tf.variable_scope('Encoder') as scope:
            
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, source)
            
            # Encoder_outpus: [max_time, batch_size, num_units]
            cell = self.create_rnn_cell(hparams)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,#CELL
                    encoder_emb_inp,#INPUTS
                    dtype=dtype,
                    sequence_length=self.batch_input.source_sequence_length,
                    time_major=self.time_major)
                
        return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_state, hparams):
        """Create decoder."""
        bos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.bos_token)), tf.int32, name = 'bos_id')
        eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.eos_token)), tf.int32, name = 'eos_id')
        start_tokens = tf.fill([self.batch_size], bos_id)
        self.end_token = eos_id
        
        # maximum_iteration: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.batch_input.source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
            
        with tf.variable_scope('Decoder') as scope:
            with tf.variable_scope("output_projection"):
                self.output_layer = layers_core.Dense(
                    self.vocab_size, use_bias=False, name="output_projection")
            
            self.decoder_cell, decoder_initial_state = self.build_decoder_cell(hparams, encoder_outputs, encoder_state, self.batch_input.source_sequence_length)
            
            # Training
            if self.training:
                
                self.target_input = self.batch_input.target_input
                if self.time_major:
                    self.target_input = tf.transpose(self.target_input)
                    
                # decoder_emp_inp: [max_time, batch_size, num_units]
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.target_input)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.batch_input.target_sequence_length, time_major=self.time_major)
                decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, decoder_initial_state,)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = self.time_major, swap_memory = True, scope = scope)

                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
                
                time_axis = 0 if self.time_major else 1
                max_time = sample_id.shape[time_axis].value or tf.shape(sample_id)[time_axis]
                
                with tf.name_scope("SampleHelper"):
                    #========================================================================================
                    # 1. Sample a Sentence, we need sent to discriminator to score it.
                    # 2. We need to score every word from generate.
                    # 3. Get every word's avg score.
                    # 4. Get Sentence score.
                    #========================================================================================
                    helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, start_tokens=start_tokens, end_token=self.end_token)
                    decoder_sample = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper_sample, decoder_initial_state,)
                    output_sample, output_sample_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder_sample, output_time_major = self.time_major, swap_memory=True, scope=scope, maximum_iterations = max_time)
                    #========================================================================================
                    # Tile our generate output.
                    # Let "I want to sleep." >> "I want to sleep." * sample_times
                    # Split every word for start token
                    # "I" > start token > "like cat."... * sample_times
                    # Concat start token become full sentence "I like cat."... * sample_times
                    # 
                    # Next loop input "want" and generate "something"...; Concat "I want" with "something"...
                    #========================================================================================
                    total_sample_times = self.sample_times * max_time
                    self.ma_time = max_time
                    self.source_sample = tf.contrib.seq2seq.tile_batch(self.batch_input.source, multiplier = total_sample_times)
                    self.source_sample_sequence_length = tf.contrib.seq2seq.tile_batch(self.batch_input.source_sequence_length, multiplier= total_sample_times)
                    #[max_time, batch_size, num_units]
                    self.target_id = tf.transpose(self.batch_input.target_output)
                    self.result_sample_id = output_sample.sample_id
                    
                    self.sample_state = tf.contrib.seq2seq.AttentionWrapperState(output_sample_state.cell_state, output_sample_state.attention, output_sample_state.time, output_sample_state.alignments, output_sample_state.alignment_history, output_sample_state.attention_state)
                    
                    with tf.name_scope("Partial_decoder"):
                        input_sample_id = tf.slice(self.result_sample_id, [0,0], [0, -1])
                        input_target_id = tf.slice(tf.transpose(self.batch_input.target_output), [0,0], [0, -1])
                        index = tf.constant(0)
                        def build_start_word(index, input_sample_id, input_target_id):
                            # get word before start word
                            word_index = tf.cast((index/self.sample_times), tf.int32)
                            max_iter = max_time - word_index
                            front_word = tf.slice(self.result_sample_id, [0,0], [tf.add(word_index,1), -1])
                            
                            if hparams.reward_type == 'MC_Search':
                                # Use Monte Carlo Search to sample words.
                                partial_start_tokens = self.result_sample_id[word_index]
                                
                                partial_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding,start_tokens = partial_start_tokens, end_token = self.end_token)
                                partial_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, partial_helper, self.sample_state,)
                                partial_output, _, _ = tf.contrib.seq2seq.dynamic_decode(partial_decoder, output_time_major = self.time_major, swap_memory=True, scope = scope, maximum_iterations = max_iter)
                                
                                target_output = tf.concat([self.target_id, tf.fill([1, self.batch_size], self.end_token)], axis = 0)
                                sample_output = tf.concat([front_word, partial_output.sample_id], axis = 0)
                            else:
                                # Let Discriminator reward partial sentence.
                                target_front_word = tf.slice(self.target_id, [0,0], [tf.add(word_index,1), -1])
                                target_output = tf.concat([target_front_word, tf.fill([max_iter, self.batch_size], self.end_token)], axis = 0)
                                sample_output = tf.concat([front_word, tf.fill([max_iter, self.batch_size], self.end_token)], axis = 0)
                            
                            output_target_id = tf.concat([input_target_id, target_output], axis=0)
                            output_sample_id = tf.concat([input_sample_id, sample_output], axis=0)
                            
                            
                            return tf.add(index, 1), output_sample_id, output_target_id
                        
                        # create loop decoder to generate [sample_times * length * batch] sentence
                        _ , output_sample_id, output_target_id = tf.while_loop(lambda index, *_: tf.less(index, total_sample_times), build_start_word,
                                                                               [index, input_sample_id, input_target_id], shape_invariants=[index.get_shape(), tf.TensorShape([None,None]), tf.TensorShape([None,None])])
                        #========================================================================================
                        # 3. Reshape output 
                        # output = [(max_time + 2) * total_sample_times , batch_size]  >> []
                        # [(max_time + 1) * total_sample_times , batch_size] > [max_time, batch_size * total_sample_times]
                        #========================================================================================
                        #shape [batch_size, length * total_sample_times]
                        self.partial_target_id = tf.transpose(output_target_id)
                        self.partial_sample_id1 = tf.transpose(output_sample_id)
                        #shape [batch_size * total_sample_times, length]
                        self.partial_target_id = tf.reshape(self.partial_target_id, [-1, tf.add(max_time,1)])
                        self.partial_sample_id3 = tf.reshape(self.partial_sample_id1, [-1, tf.add(max_time,1)])
                        if hparams.reward_type == 'MC_Search':
                            self.partial_sample_id = tf.slice(self.partial_sample_id3, [0,0], [-1, tf.add(max_time,-1)])
                            self.partial_sample_id = tf.concat([self.partial_sample_id, tf.fill([tf.shape(self.partial_sample_id)[0], 1], self.end_token)], axis=1)
                        #shape [length, batch_size * total_sample_times]
                        self.partial_sample_id = tf.transpose(self.partial_sample_id)
            # Inference
            else:
                
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                
                if beam_width > 0:
                    with tf.name_scope("BeamSearchDecoder"):
                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=self.decoder_cell,
                            embedding=self.embedding,
                            start_tokens=start_tokens,
                            end_token=self.end_token,
                            initial_state=decoder_initial_state,
                            beam_width=beam_width,
                            output_layer=self.output_layer,
                            length_penalty_weight=length_penalty_weight)
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding, start_tokens, self.end_token)
                    decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, decoder_initial_state, output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=maximum_iterations, output_time_major=self.time_major, 
                    swap_memory=True, scope=scope)

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id
                    
        return logits, sample_id, final_context_state
                    
        
    def build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                           source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        num_units = hparams.num_units
        beam_width = hparams.beam_width

        dtype = tf.float32
        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs
        
        if not self.training and beam_width > 0:
            with tf.name_scope('Tile_batch'):
                memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
                source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,multiplier=beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size
        
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
        
        cell = self.create_rnn_cell(hparams)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (not self.training and beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=num_units, alignment_history=alignment_history, name="attention")

        if hparams.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state
    
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