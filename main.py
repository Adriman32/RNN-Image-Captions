import string
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.callbacks import TensorBoard
import pickle
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def create_caption_dict(dict_file):
    desc_dict = {}
    file = open(dict_file, 'r')
    text = file.read()
    for line in text.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in desc_dict:
            desc_dict[image_id] = list()
        desc_dict[image_id].append(image_desc)
    return desc_dict

def clean_captions(caption_dict):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in caption_dict.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(['startseq']+desc+['endseq'])
    return caption_dict

def create_vocabulary(caption_dict,word_count_threshold = 10):
    vocabulary = set()
    for key in caption_dict.keys():
        [vocabulary.update(d.split()) for d in caption_dict[key]]

    training_captions = []
    word_counts = {}
    
    for key, desc_list in caption_dict.items():
        for desc in desc_list:
            training_captions.append(desc)
    count = 0
    for sent in training_captions:
        count += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocabulary = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    vocabulary.append('0')
    return vocabulary


def load_dataset(filename):
    file = open(filename, 'r')
    text = file.read()
    dataset = []
    for line in text.split('\n'):
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_descriptions(data_set, caption_dict):
    train_descriptions = {}
    for image_id in data_set:
        if image_id in caption_dict:
            train_descriptions[image_id] = caption_dict[image_id]
    return train_descriptions

def load_images(filepath):
    img_dict = {}
    for filename in os.listdir(filepath):
        img_name = filename
        img = load_img(filepath+filename, target_size=(299, 299))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img_dict[img_name] = img
    return img_dict

def encode_images(img_dict):
    model = InceptionV3(weights='imagenet')
    model = tf.keras.Model(model.input, model.layers[-2].output)
    feature_dict = {}
    for img_name in img_dict:
        img = img_dict[img_name]
        feature = model.predict(img, verbose=0)
        feature_dict[img_name] = feature

    return feature_dict
    
def save_image_features(feature_dict, filename):
    if not(filename.endswith('.pkl')):
        filename = filename + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calc_max_length(caption_dict):
    max_length = 0
    for key in caption_dict.keys():
        for cap in caption_dict[key]:
            max_length = max(max_length, len(cap.split()))
    return max_length

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch, num_classes):
    X1, X2, y = list(), list(), list()
    n=0
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            photo = photos[key+'.jpg']
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=num_classes)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n==num_photos_per_batch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0

def create_word_embeddings(glove_filepath, vocab, wordtoix ,embedding_dim):
    glove_file = open(glove_filepath, 'r', encoding="utf8")
    glove_dict = {}
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_dict[word] = coefs
    glove_file.close()

    embedding_dim = 200
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in wordtoix.items():
        if i < len(vocab):
            embedding_vector = glove_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_model(vocab_size, max_length, embedding_matrix):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 200, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    return model

def train_model(model, train_dict, generator):
    epochs = 20
    steps = len(train_dict)
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    for i in range(epochs):
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[tensorboard_callback])
        model.save('output_models/new/model_epoch_' + str(i) + '.h5')
    return model

def predict(model, image, wordtoix, max_length):
    in_text = 'startseq'
    caption = ''
    for _ in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([image, sequence], verbose=0)
        pred = np.argmax(pred)
        word = ixtoword.get(pred, '')
        if word == 'endseq':
            break
        in_text += ' ' + word
    caption = in_text.strip()
    return caption

def evaluate_model(model, test_dict, enc_imgs, wordtoix, max_length):
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for key, desc_list in test_dict.items():
        photo = enc_imgs[key + '.jpg']
        photo = photo.reshape((1, 2048))
        pred = predict(model, photo, wordtoix, max_length)
       
        for reference in desc_list:
            smoothing_function = SmoothingFunction().method2
            bleu1 = sentence_bleu(reference.split(), pred.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu2 = sentence_bleu(reference.split(), pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
            bleu3 = sentence_bleu(reference.split(), pred.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
            bleu4 = sentence_bleu(reference.split(), pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
                   
    # Calculate average BLEU scores
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores)
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
    
    return avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4

                   
if __name__ == '__main__':
    caption_dict = create_caption_dict('D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/Flickr_8k_Text/Flickr_8k.token.txt')
    clean_caption_dict = clean_captions(caption_dict)
    vocab = create_vocabulary(clean_caption_dict)
    

    train_file = 'D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/Flickr_8k_Text/Flickr_8k.trainImages.txt'
    test_file = 'D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/Flickr_8k_Text/Flickr_8k.testImages.txt'

    train_set = load_dataset(train_file)
    test_set = load_dataset(test_file)

    train_dict = load_descriptions(train_set, clean_caption_dict)
    test_dict = load_descriptions(test_set, clean_caption_dict)

    enc_imgs =  open('D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/encoded_images.pkl', 'rb')
    enc_imgs   = pickle.load(enc_imgs)
    enc_imgs = {k: np.squeeze(v) for k, v in enc_imgs.items()}
        
    
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    max_desc_length = calc_max_length(train_dict)

    # Create data generator
    generator = data_generator(train_dict, enc_imgs, wordtoix, max_desc_length, 3, len(vocab))
    word_embeddings = create_word_embeddings('D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/glove.6B.200d.txt', vocab, wordtoix, 200)

    # Create model
    model = create_model(len(vocab), max_desc_length, word_embeddings)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    trained_model = train_model(model, train_dict, generator)


    model = tf.keras.models.load_model("output_models/new/model_epoch_19.h5")

    # Select a random image
    image_id = np.random.choice(list(test_dict.keys()))
    image = enc_imgs[image_id + ".jpg"]
    image = image.reshape((1, 2048))

    # Greedy
    greedy_caption = predict(model, image, wordtoix, max_desc_length)
    greedy_caption = greedy_caption.replace("startseq ", "")

    # Plot image with captions
    plt.imshow(load_img(f"D:/Users/Adrian Ruvalcaba/Documents/School/2022-2023/Spring 2023/ECE 6524/Projects/Project 3/Flickr_8k_Images/{image_id + '.jpg'}"))
    plt.axis('off')

    plt.title(f"Greedy:\n{greedy_caption}\n\nTrue:\n{test_dict[image_id][0][9:-7]}",fontsize=8)

    plt.subplots_adjust(hspace=0.4, wspace=0.4) 
    plt.savefig('output.png', bbox_inches='tight', dpi=300)
    plt.show()

    print("Done")
















    
