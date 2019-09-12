import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Reshape, LSTM, Lambda, Dropout, add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model

from config import text_path, img_w, img_h, batch_size, epochs
from batch_generator import words, get_text, batch_generator, max_length, letters


def evaluate(input_model):

    # Прямое распространение, декодирование выходов сети и расчет точности
    correct_prediction = 0
    generator = batch_generator(words)
    inputs, outputs = next(generator)

    x_test = inputs['the_input']
    y_test = inputs['the_labels']
    y_pred = input_model.predict(x_test)
    shape = y_pred[:, 2:, :].shape

    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :max_length]

    for m in np.arange(batch_size):
        text_pred = ''.join([letters[k] for k in out[m]])
        text_pred = text_pred.replace('-', '')
        text_true = ''.join([letters[k] for k in y_test[m]])
        text_true = text_true.replace('-', '')

        if text_pred == y_test[m]:
            correct_prediction += 1
        else:
            print('Text pred:\t{}'.format(text_pred))
            print('Text true:\t{}\n'.format(text_true))

    return correct_prediction / 100

class Evaluate(Callback):

    def on_epoch_end(self, epoch, logs=None):

        # Вывод точности в конце каждой эпохи
        acc = evaluate(base_model)
        print('Accuracy:t\' + str(acc) + "%")

evaluator = Evaluate()

def ctc_lambda_func(args):
              
    iy_pred, ilabels, iinput_length, ilabel_length = args
    iy_pred = iy_pred[:, 2:, :]
              
    return K.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)

# Построение модели
inputShape = Input(name='the_input', shape=(img_w, img_h, 1))

# CNN
conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputShape)
batchnorm_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D(pool_size=(2, 2))(batchnorm_1)
conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_2)
batchnorm_2 = BatchNormalization()(conv_3)

bn_shape = batchnorm_2.get_shape()

# Изменение формы перед подачей на LSTM
x_reshape = Reshape(target_shape=(int(int(bn_shape[1]) * 2), int(int(bn_shape[2] * bn_shape[3])/2)))(batchnorm_2)

# 1-й полносвязный слой
fc_1 = Dense(64, activation='relu')(x_reshape)

# Двунаправленные LSTM-слои
rnn_1 = LSTM(64, kernel_initializer='he_normal', return_sequences=True)(fc_1)
rnn_1b = LSTM(64, kernel_initializer='he_normal', go_backwards=True, return_sequences=True)(fc_1)
rnn1_merged = add([rnn_1, rnn_1b])
rnn_2 = LSTM(64, kernel_initializer='he_normal', return_sequences=True)(rnn1_merged)
rnn_2b = LSTM(64, kernel_initializer='he_normal', go_backwards=True, return_sequences=True)(rnn1_merged)
rnn2_merged = concatenate([rnn_2, rnn_2b])

# Регуляризация и выходной слой с softmax
drop_1 = Dropout(0.25)(rnn2_merged)
fc_2 = Dense(len(letters), kernel_initializer='he_normal', activation='softmax')(drop_1)

# Модель для прелсказаний
base_model = Model(inputs=[inputShape], outputs=[fc_2])

labels = Input(name='the_labels', shape=[max_length], dtype=np.float32)
input_length = Input(name='input_length', shape=[1], dtype=np.int32)
label_length = Input(name='label_length', shape=[1], dtype=np.int32)

# Lambda-слой CTC-loss
loss_out = Lambda(ctc_lambda_func, output_shape=[1], name='ctc')([fc_2, labels, input_length, label_length])

# Модель для обучения
model = Model(inputs=[inputShape, labels, input_length, label_length], outputs=loss_out)

# Функция ошибки - CTC-loss, оптимизатор - Adam
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(), metrics=['acc'])
model.summary()

plot_model(model, to_file='model.png', show_shapes=True)

model.fit_generator(batch_generator(words), steps_per_epoch=words.size // batch_size, epochs=epochs, verbose=1,  callbacks=[evaluator])
base_model.save('base_model.h5')
