import pandas as pd
import numpy as np
import cairo
import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from config import text_path, img_w, img_h, batch_size

def get_text(path):

	with open(path, 'r') as f:

		# Считываем файл как строку
		words_str = f.read()

		# Разделяем строку на слова и преобразуем в pd.Series
		words = pd.Series(words_str.replace('\n', ' ').split(' '))

		# Убираем лишние строки
		words = words[(words != '<S>') & (words != '')]
		words.index = pd.Index(np.arange(words.size) + 1)

		return words

def random_move(context, text, img_h, img_w):

	# Получаем размеры входного текста
	x, y, width, height, dx, dy = context.text_extents(text)

	# Вычисляем случайные координаты текста
	rand_x = np.random.choice(np.arange(img_w - width))
	rand_y = np.random.choice(np.arange(height, img_h - 1))

	return rand_x, rand_y

def paint_text(text, img_h, img_w, position='random', font_size = 12):

    # Создаем поверхность и контекст cairo
    imsurf = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_w, img_h)
    context = cairo.Context(imsurf)

    # Указываем для контекста черные чернила
    context.set_source_rgb(0, 0, 0)

    # Выбор типа и размера шрифта
    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size)

    # Получаем размеры входного текста, исключая слишком большие значения
    x, y, width, height, dx, dy = context.text_extents(text)
    assert (width < img_w) and (height < img_h - 1)

    # Располагаем текст на поверхности
    if position == 'random':
        r_x, r_y = random_move(context, text, img_h, img_w)
        context.move_to(r_x, r_y)
    elif position == 'center':
        context.move_to(img_w/2 - width/2, img_h/2 + height/2)

    context.show_text(text)

    # На выходе матрица numpy формы (img_h, img_w)
    buf = imsurf.get_data()
    img = np.frombuffer(buf, np.uint8)
    img.shape = (img_h, img_w, 4)

    return img[:, :, 3]

def add_normal_distr(img, std=10):
	
	normal_distr = np.random.normal(size=(img.shape[0], img.shape[1], 1), scale=std)
	img_norm = img.astype(np.int16) + normal_distr.astype(np.int16)
	img_norm[np.where(img_norm < 0)] = 0
	img_norm[np.where(img_norm > 255)] = 255

	return img_norm.astype(img.dtype)

def get_max_lenght(dataset):
	return max([len(dataset[i]) for i in dataset.index])

def get_letters(dataset):
	return list(pd.Series([letter for word in words for letter in word]).unique()) + ['-']

def text_to_labels(letters, text):
        return [letters.index(x) for x in text if x in letters]

# По словарю определяем максимальную длину слова и т.н. алфавит (набор уникальных символов, встречаемых в словаре)
words = get_text(text_path)
max_length = get_max_lenght(words)
letters = get_letters(words)

def batch_generator(dataset, batch_size = batch_size):

    epoch = 0

    for i in np.arange(10000000):
        
        x_data = np.zeros((batch_size, img_h, img_w, 1))
        y_data = np.zeros((batch_size, max_length), dtype = 'int32')
        
        for j in np.arange(batch_size) + 1:

        	try:
	            ind = i * batch_size + j - dataset.size * epoch

	            if ind >= dataset.size:
	                epoch += 1

	            # Генерируем зашумленное изображение с текстом, расположенном в случайном месте
	            img = paint_text(dataset[ind], img_h, img_w, position='random')[:, :, np.newaxis]
	            img = add_normal_distr(img, std=30)
	            text = dataset[ind]

	            # Дополняем строчку до единого размера max_length
	            while len(text) < max_length:
	                text += '-'

	            # Преобразуем текст в массив NumPy 
	            lbl = np.asarray(text_to_labels(letters, text), dtype = np.int32)

	            x_data[j - 1, :, :, :] = img
	            y_data[j - 1, :] = lbl
	            inputs = {'the_input': x_data,
	                        'the_labels': y_data,
	                        'input_length': np.ones(batch_size) * int(img.shape[1] - 2),
	                        'label_length': np.ones(batch_size) * max_length}
	            outputs = {'ctc': np.zeros([batch_size])}

	        except AssertionError:
	        	continue

        yield inputs, outputs

if __name__ == '__main__':
	
	test_img = paint_text('figure', 64, 64, position='random', font_size = 16)
	test_img = add_normal_distr(test_img, std=30.0)
	test_img = 255 - test_img
	plt.imshow(test_img, cmap='gray')
	plt.show()
	plt.plot()
