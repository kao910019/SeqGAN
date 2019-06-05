# -*- coding: utf-8 -*-
import codecs
import os
import tensorflow as tf
import sys

from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd

from collections import namedtuple
from tensorflow.python.ops import lookup_ops

sys.path.append("..")
from settings import SYSTEM_ROOT, CORPUS_DIR, VOCAB_FILE, MODEL_FILE
from settings import TRAIN_FILE, TEST_FILE

#COMMENT_LINE_STT = "#=="
#CONVERSATION_SEP = "==="

AUG0_FOLDER = "Augment0"
AUG1_FOLDER = "Augment1"
AUG2_FOLDER = "Augment2"

MAX_LEN = 1000  # Assume no line in the training data is having more than this number of characters

class TokenizedData:
    def __init__(self, hparams=None, training=True, buffer_size=8192):
        
        self.training = training
        self.hparams = hparams

        self.src_max_len = self.hparams.src_max_len
        self.tgt_max_len = self.hparams.tgt_max_len
        
        self.vocab_size, self.vocab_list = check_vocab(VOCAB_FILE)
        self.vocab_table = lookup_ops.index_table_from_file(VOCAB_FILE,
                                                            default_value=self.hparams.unk_id)
        
        if os.path.isfile(MODEL_FILE):
            print("# Load Word2Vec model")
            self.embedding_model = Word2Vec.load(MODEL_FILE)
        else:
            print("# Word2Vec model doesn't exist")
            self.embedding_model = self.create_embedding(CORPUS_DIR, self.hparams.embedding_size)
            print("# Save Word2Vec model")
            self.embedding_model.save(MODEL_FILE)
        
        if training:
            self.case_table = prepare_case_table()
#            self.reverse_vocab_table = None
            
            self.reverse_vocab_table = \
                lookup_ops.index_to_string_table_from_file(VOCAB_FILE,
                                                           default_value=self.hparams.unk_token)
                
            train_text_set = self._load_corpus(TRAIN_FILE)
            test_text_set = self._load_corpus(TEST_FILE)
            self.train_id_set = self._convert_to_tokens(buffer_size, train_text_set)
            self.test_id_set = self._convert_to_tokens(buffer_size, test_text_set)
            
#            self.text_set = self._load_corpus(CORPUS_DIR)
#            self.id_set = self._convert_to_tokens(buffer_size, self.text_set)
            
        else:
            self.case_table = None
            self.reverse_vocab_table = \
                lookup_ops.index_to_string_table_from_file(VOCAB_FILE,
                                                           default_value=self.hparams.unk_token)
        
    def get_training_batch(self, id_set, num_threads=4):
        
        with tf.name_scope("make_training_batch"):
            
            assert self.training
            #256 * 400
            buffer_size = self.hparams.batch_size# * 400
    
            # Comment this line for debugging.
            dataset = id_set.shuffle(buffer_size=buffer_size)
            
            # Create a target input prefixed with BOS and a target output suffixed with EOS.
            # After this mapping, each element in the dataset contains 3 columns/items.
            dataset = dataset.map(lambda src, tgt:
                                      (src,
                                       tf.concat(([self.hparams.bos_id], tgt), 0),
                                       tf.concat((tgt, [self.hparams.eos_id]), 0),
                                       ),num_parallel_calls=num_threads).prefetch(buffer_size)
            
            # Add in sequence lengths.
            dataset = dataset.map(lambda src, tgt_in, tgt_out:
                                      (src, 
                                       tgt_in, tgt_out,
                                       tf.size(src), tf.size(tgt_in)
                                       ),num_parallel_calls=num_threads).prefetch(buffer_size)
    
            def batching_func(x):
                return x.padded_batch(
                    self.hparams.batch_size,
                    # The first three entries are the source and target line rows, these have unknown-length
                    # vectors. The last two entries are the source and target row sizes, which are scalars.
                    padded_shapes=(tf.TensorShape([None]),  # src
                                   tf.TensorShape([None]),  # tgt_input
                                   tf.TensorShape([None]),  # tgt_output
                                   tf.TensorShape([]),      # src_len
                                   tf.TensorShape([])),     # tgt_len
                    # Pad the source and target sequences with eos tokens. Though we don't generally need to
                    # do this since later on we will be masking out calculations past the true sequence.
                    padding_values=(self.hparams.eos_id,  # src
                                    self.hparams.eos_id,  # tgt_input
                                    self.hparams.eos_id,  # tgt_output
                                    0,       # src_len -- unused
                                    0))      # tgt_len -- unused
    
#            if self.hparams.num_buckets > 1:
#                bucket_width = (self.src_max_len + self.hparams.num_buckets - 1) // self.hparams.num_buckets
#    
#                # Parameters match the columns in each element of the dataset.
#                def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
#                    # Calculate bucket_width by maximum source sequence length. Pairs with length [0, bucket_width)
#                    # go to bucket 0, length [bucket_width, 2 * bucket_width) go to bucket 1, etc. Pairs with
#                    # length over ((num_bucket-1) * bucket_width) words all go into the last bucket.
#                    # Bucket sentence pairs by the length of their source sentence and target sentence.
#                    bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
#                    return tf.to_int64(tf.minimum(self.hparams.num_buckets, bucket_id))
#    
#                # No key to filter the dataset. Therefore the key is unused.
#                def reduce_func(unused_key, windowed_data):
#                    return batching_func(windowed_data)
#    
#                batched_dataset = dataset.apply(
#                    tf.contrib.data.group_by_window(key_func=key_func,
#                                                    reduce_func=reduce_func,
#                                                    window_size=self.hparams.batch_size))
#            else:
            batched_dataset = batching_func(dataset)
        
            iterator = batched_dataset.make_initializable_iterator()
#            (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (iterator.get_next())
            
            return BatchedInput(iterator=iterator,
                                batched_dataset=batched_dataset,
                                handle=iterator.string_handle(),
                                initializer=iterator.initializer,
                                source=None,                         #Quention sentence id
                                target_input=None,             #Answer sentence id, teacher forcing
                                target_output=None,           #Answer sentence id, compute cross entropy 
                                source_sequence_length=None,
                                target_sequence_length=None)

    def get_inference_batch(self, src_dataset):
        
        with tf.name_scope("make_inference_batch"):
            
            text_dataset = src_dataset.map(lambda src: (tf.string_split([src]).values))
            if self.hparams.src_max_len_infer:
                text_dataset = text_dataset.map(lambda src: (src[:self.hparams.src_max_len_infer]))
            id_dataset = text_dataset.map(lambda src: (tf.cast(self.vocab_table.lookup(src),tf.int32)))
            if self.hparams.source_reverse:
                id_dataset = id_dataset.map(lambda src: (tf.reverse(src, axis=[0])))
            id_dataset = id_dataset.map(lambda src: (src, tf.size(src)))
            
            def batching_func(x):
                return x.padded_batch(self.hparams.batch_size_infer,
                                      padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])),
                                      padding_values=(self.hparams.eos_id,0))
            
            batched_dataset = batching_func(id_dataset)
            iterator = batched_dataset.make_initializable_iterator()
            (src_ids, src_seq_len) = iterator.get_next()
            
        return BatchedInput(iterator=iterator,
                            batched_dataset=batched_dataset,
                            handle=iterator.string_handle(),
                            initializer=iterator.initializer,
                            source=src_ids,
                            target_input=None,
                            target_output=None,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=None)
        
    def multiple_batch(self, handler, dataset):
        """Make Iterator switch to change batch input"""
        iterator = tf.data.Iterator.from_string_handle(handler, dataset.output_types, dataset.output_shapes)
        (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (iterator.get_next())
        return BatchedInput(iterator=None,
                            batched_dataset=None,
                            handle=None,
                            initializer=None,
                            source=src_ids,
                            target_input=tgt_input_ids,
                            target_output=tgt_output_ids,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=tgt_seq_len)
        
#    def _load_corpus(self, corpus_dir):
#        text_set = None
#            
#        for fd in range(1):
#            file_list = []
#            if fd == 0:
#                file_dir = os.path.join(corpus_dir, 'Data', AUG0_FOLDER)
#            elif fd == 1:
#                file_dir = os.path.join(corpus_dir, 'Data', AUG1_FOLDER)
#            else:
#                file_dir = os.path.join(corpus_dir, 'Data', AUG2_FOLDER)
#                
#            for data_file in sorted(os.listdir(file_dir)):
#                full_path_name = os.path.join(file_dir, data_file)
#                if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
#                    file_list.append(full_path_name)
#                    
#            assert len(file_list) > 0
#            
#            dataset = tf.data.TextLineDataset(file_list)
#                
#            with tf.name_scope("load_corpus"):
#                src_dataset = dataset.filter(lambda line:
#                                             tf.logical_and(tf.size(line) > 0,
#                                                            tf.equal(tf.substr(line, 0, 2), tf.constant('Q:'))))
#                src_dataset = src_dataset.map(lambda line:
#                                              tf.substr(line, 2, MAX_LEN)).prefetch(4096)
#                tgt_dataset = dataset.filter(lambda line:
#                                             tf.logical_and(tf.size(line) > 0,
#                                                            tf.equal(tf.substr(line, 0, 2), tf.constant('A:'))))
#                tgt_dataset = tgt_dataset.map(lambda line:
#                                              tf.substr(line, 2, MAX_LEN)).prefetch(4096)
#                src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
#                
#                if fd == 1:
#                    src_tgt_dataset = src_tgt_dataset.repeat(self.hparams.aug1_repeat_times)
#                elif fd == 2:
#                    src_tgt_dataset = src_tgt_dataset.repeat(self.hparams.aug2_repeat_times)
#    
#                if text_set is None:
#                    text_set = src_tgt_dataset
#                else:
#                    text_set = text_set.concatenate(src_tgt_dataset)
#                
#        return text_set
    
    def _load_corpus(self, file):
        
        dataset = tf.data.TextLineDataset(file)
        
        with tf.name_scope("load_corpus"):
            src_dataset = dataset.filter(lambda line: tf.logical_and(tf.size(line) > 0, tf.equal(tf.substr(line, 0, 2), tf.constant('Q:'))))
            src_dataset = src_dataset.map(lambda line: tf.substr(line, 2, MAX_LEN)).prefetch(4096)
            tgt_dataset = dataset.filter(lambda line: tf.logical_and(tf.size(line) > 0, tf.equal(tf.substr(line, 0, 2), tf.constant('A:'))))
            tgt_dataset = tgt_dataset.map(lambda line: tf.substr(line, 2, MAX_LEN)).prefetch(4096)
            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
            
        return src_tgt_dataset
    
    def _convert_to_tokens(self, buffer_size, text_set):
        # The following 3 steps act as a python String lower() function
        # Split to characters
        
        with tf.name_scope("convert_to_tokens"):
            
            text_set = text_set.map(lambda src, tgt:
                                    (tf.string_split([src], delimiter='').values,
                                     tf.string_split([tgt], delimiter='').values)).prefetch(buffer_size)
            # Convert all upper case characters to lower case characters
            text_set = text_set.map(lambda src, tgt:
                                    (self.case_table.lookup(src),
                                     self.case_table.lookup(tgt))).prefetch(buffer_size)
            # Join characters back to strings
            text_set = text_set.map(lambda src, tgt:
                                    (tf.reduce_join([src]),
                                     tf.reduce_join([tgt]))).prefetch(buffer_size)
    
            # Split to word tokens
            text_set = text_set.map(lambda src, tgt:
                                    (tf.string_split([src]).values, 
                                     tf.string_split([tgt]).values)).prefetch(buffer_size)
                
            # Remove sentences longer than the model allows
            text_set = text_set.map(lambda src, tgt:
                                    (src[:self.src_max_len],
                                     tgt[:self.tgt_max_len])).prefetch(buffer_size)
    
            # Reverse the source sentence if applicable
            if self.hparams.source_reverse:
                text_set = text_set.map(lambda src, tgt:
                                        (tf.reverse(src, axis=[0]), tgt)
                                        ).prefetch(buffer_size)
    
            # Convert the word strings to ids.  Word strings that are not in the vocab get
            # the lookup table's default_value integer.
            id_set = text_set.map(lambda src, tgt:
                                  (tf.cast(self.vocab_table.lookup(src), tf.int32),
                                   tf.cast(self.vocab_table.lookup(tgt), tf.int32))).prefetch(buffer_size)
        return id_set
    
#========================================================================================
                                        
    def load_sentence(self, sen_file):
        text_list = []
        with open(sen_file,'r',encoding = 'utf8') as f:
            name = sen_file.split("/")[-1]
            for line in tqdm(f, desc = name):
                if line.startswith("\ufeff"):
                    line = line[1:]
                text_list.append(line.strip())
        return text_list
    
    def corpus_embedding_series(self, corpus_dir):
        
        sent_series = None
        file_list = []
        for fd in range(2, -1, -1):
            if fd == 0:
                file_dir = os.path.join(corpus_dir, 'Data', AUG0_FOLDER)
            elif fd == 1:
                file_dir = os.path.join(corpus_dir, 'Data', AUG1_FOLDER)
            else:
                file_dir = os.path.join(corpus_dir, 'Data', AUG2_FOLDER)
            for data_file in sorted(os.listdir(file_dir)):
                full_path_name = os.path.join(file_dir, data_file)
                if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                    file_list.append(full_path_name)
                    
        assert len(file_list) > 0
        for file in file_list:
            if sent_series:
                sent_series += self.load_sentence(file)
            else:
                sent_series = self.load_sentence(file)
                    
        return pd.Series(sent_series)
    
    def create_embedding(self, corpus_dir, embedding_size):
        
        print("# Load corpus")
        src_list = self.corpus_embedding_series(corpus_dir)
        QQ = ((src_list[src_list.str.startswith("Q: ")]).str.replace("Q: ","")).str.split(" ")
        AA = ((src_list[src_list.str.startswith("A: ")]).str.replace("A: ","")).str.split(" ")
        corpus = pd.concat([QQ, AA]).sample(frac=1)
        print("# Create a new Word2Vec model from corpus")
        embedding_model = Word2Vec(corpus, size = embedding_size)
        print("# Word2Vec vocab size =",len(embedding_model.wv.vocab))
        return embedding_model
    
    def write_vocab(self, file_path, vocab_list):
        self.vocab_list += vocab_list
        if tf.gfile.Exists(file_path):
            with open(file_path,'w',encoding = 'utf8') as f:
                for text in self.vocab_list:
                    f.write("{}\n".format(text))
        else:
            raise ValueError("The vocab_file does not exist. Please run the vocab_generator.py to create it.")
        return check_vocab(file_path)
        
    
#========================================================================================
        
def check_vocab(file_path):
    if tf.gfile.Exists(file_path):
        vocab_list = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(file_path, "rb")) as f:
            for word in f:
                vocab_list.append(word.strip())
    else:
        raise ValueError("The vocab_file does not exist. Please run the vocab_generator.py to create it.")

    return len(vocab_list), vocab_list

def prepare_case_table():
    keys = tf.constant([chr(i) for i in range(32, 127)])

    l1 = [chr(i) for i in range(32, 65)]
    l2 = [chr(i) for i in range(97, 123)]
    l3 = [chr(i) for i in range(91, 127)]
    values = tf.constant(l1 + l2 + l3)

    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), ' ')

class BatchedInput(namedtuple("BatchedInput",
                              ["iterator",
                               "batched_dataset",
                               "handle",
                               "initializer",
                               "source",
                               "target_input",
                               "target_output",
                               "source_sequence_length",
                               "target_sequence_length"])):
    pass
    