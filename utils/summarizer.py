from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import numpy as np 
from utils.glove_loader import GLOVE_EMBEDDING_SIZE, load_cur_glove



class Seq2seqGloveSummarizer(object):

    model_name = 'seq2seq-glove'

    def __init__(self, config):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        # seq2seq model config/define
        
        # encoder part 
        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=128, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        # decoder part 
        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=128, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        '''
        Adding Dense layers into the decoder
        softmax we don't use glove embedding into the dense layer
        make it as a classification problem
        '''
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        '''
        According to the keras document 

        Optimizer 'rmsprop' that implements the rmsprop algorithm
        the gist of rmsprop is to：
        - Maintain a moving (discounted) average of the square of gradients
        - Divide the gradient by the root of this average
        '''
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model 
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_inputs = [Input(shape=(128, )), Input(shape=(128,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_glove(self, embedding_path):
        self.word2em = load_cur_glove(embedding_path)

    
    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, GLOVE_EMBEDDING_SIZE))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0
                        if w in self.word2em:
                            emb = self.unknown_emb
                            decoder_input_data_batch[lineIdx, idx, :] = emb
                        if w in self.target_idx2word:    
                                w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx-1, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx-1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch
    
    
    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break 
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp
    
    
    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp, dtype=object)
        print(temp.shape)
        return temp

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2seqGloveSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2seqGloveSummarizer.model_name + '-weight.h5'


    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2seqGloveSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xdev, Ydev, epochs=None, batch_size=None, model_dir_path=None):
        if epochs == None:
            epochs = 10
        if batch_size == None:
            batch_size = 64
        if model_dir_path == None:
            model_dir_path = './model'

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2seqGloveSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2seqGloveSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2seqGloveSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ydev = self.transform_target_encoding(Ydev)

        Xtrain = self.transform_input_text(Xtrain)
        Xdev = self.transform_input_text(Xdev)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        dev_gen = self.generate_batch(Xdev, Ydev, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        dev_num_batches = len(Xdev) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=True,
                                           validation_data=dev_gen, validation_steps=dev_num_batches,
                                           callbacks=[checkpoint])

        self.model.save_weights(weight_file_path)
        return history

 
