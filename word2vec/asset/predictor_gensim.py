import pandas as pd
import numpy as np

from gensim.models import Word2Vec


def cosine_distance(model, word, target_list, num):
    """
    Computes the cosine similarity based on the angle between two vectors.
    Differs from Euclidian Distance which is based on A² + B² = C² (not suited for high dimensional vector due to "automatic" increase of Euclidian Sim for increase in dims)

    sim(A,B) = cos(theta) = (A*B)/(norm(A)*norm(B))

    norm = length of vector
    :param model:
    :param word:
    :param target_list:
    :param num:
    :return:
    """
    cosine_dict = {}
    word_list = []
    a = model[word]

    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1], reverse=True)  ## in Descedning order

    for item in dist_sort:
        word_list.append((item[0], item[1]))

    return word_list[0:num]


# import data
data = pd.read_csv('word2vec/data/car_features/data.csv')

# format data: gensim requires list of list
data['Maker_Model'] = data['Make'] + " " + data['Model']

# select features
df = data[['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style',
           'Maker_Model']]

# combine all columns into one column for each row
df = df.apply(lambda x: ','.join(x.astype(str)), axis=1)
df_clean = pd.DataFrame({'clean': df})

# create list of lists
sent = [row.split(',') for row in df_clean['clean']]

# train own gensim model
"""
min_count:  minimum count of words to consider when training model; words with less occurrence will be ignored
size:       number of dimensions
workers:    partitions during training
sg:         training algo (skip-gram = 1)
"""
model = Word2Vec(sent, min_count=1, size=50, workers=3, window=3, sg=1)

# get similarities (euclidian distance)
model.similarity('Porsche 718 Cayman', 'Nissan Van')
model.most_similar('Mercedes-Benz SLK-Class')[:5]

# Cosine similarities
Maker_Model = list(data.Maker_Model.unique())
cosine_distance(model, 'Mercedes-Benz SLK-Class', Maker_Model, 5)


