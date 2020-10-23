"""
We are going to train the Neural Net to do the following:
Given a specific word in the middle of a sentence (input word), look at the words nearby and pick one at random. The network is telling us the probability for every word in our
vocabulary of being the "nearby word" that we chose.
The output probabilities are going to relate to how likely it is to find each vocabulary word nearby our input word.

We are going to represent the input word as a one-hot vector (one component for every word in our vocab). A 1 in the position of the corresponding word and 0s in all other.
The output of the network is a single vector (one component for every word) containing the probability that a randomly selected nearby word is that vocabulary word.
Thus, the hidden layer is basically operating as a lookup table since it will only select the matrix row corresponding to the 1 in the input vector.

Ultimately, if two words have similar contexts (words that are likely to appear around them), our model needs to output similar results for these two words.
"""

"""
To-Dos:

- Tokenizer
- Remove input word for vec_sim from list
- Train data as func
- Refactor into OOP standards
"""

import numpy as np

from collections import defaultdict


def word2onehot(word, vocab_length, word_index):
    # Initialise a blank vector
    word_vec = np.zeros(vocab_length)

    # Get ID of word from word_index
    word_id = word_index[word]

    # Change the value from 0 to 1 at the target word's position
    word_vec[word_id] = 1

    return word_vec


def forward_pass(target_vector, weight_matrix1, weight_matrix2):
    # x is one-hot encoded vector, shape (nr_unique_words)*1 (9x1)
    # Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 returns 10x1
    h = np.dot(target_vector, weight_matrix1)
    # Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1 - one for each unique_word
    u = np.dot(h, weight_matrix2)
    # Run 1x9 through softmax to force each element to range of [0, 1] (probabilites) - 1x8
    y_c = softmax(u)

    return y_c, h, u


def softmax(target_vector):
    e_x = np.exp(target_vector - np.max(target_vector))
    softmaxed = e_x / e_x.sum(axis=0)

    return softmaxed


def backprop(error, hidden_layer, target_vector, learning_rate, weight_matrix1, weight_matrix2):
    # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
    # Going backward, we need to take the derivative of E with respect to w2
    # h - shape 10x1, e - shape 9x1, d1_dw2 - shape 10x9
    # x shape 9x1, w2 - shape 10x9, e.T - shape 9x1
    dl_dw2 = np.outer(hidden_layer, error)
    dl_dw1 = np.outer(target_vector, np.dot(w2, error.T))

    #########################################
    # print('Delta for w2', dl_dw2)			#
    # print('Hidden layer', h)				#
    # print('np.dot', np.dot(w2, e.T))	    #
    # print('Delta for w1', dl_dw1)			#
    #########################################

    # Update weights
    weight_matrix1 = weight_matrix1 - (learning_rate * dl_dw1)
    weight_matrix2 = weight_matrix2 - (learning_rate * dl_dw2)


def word_vec(word, word_index):
    w_index = word_index[word]
    v_w = w1[w_index]

    return v_w


def vec_sim(word, weight_matrix1, top_n, index_word, word_index):
    v_w1 = word_vec(word, word_index)
    word_sim = {}

    for i in range(nr_unique_words):
        # Find the similarity score for each word in the vocab
        v_w2 = weight_matrix1[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den

        word = index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

    for word, sim in words_sorted[:top_n]:
        print(word, sim)


# Initialise text
# Include basic tokenizer func and to lower)
text = "natural language processing and machine learning is fun and exciting"
corpus = [[word.lower() for word in text.split()]]

word_counts = defaultdict(int)
for row in corpus:
    for word in row:
        word_counts[word] += 1

# Vocab length
nr_unique_words = len(word_counts.keys())

# Generate Lookup Dictionaries (vocab)
words_list = list(word_counts.keys())

# Generate word:index
word_index = dict((word, i) for i, word in enumerate(words_list))

# Generate index:word
index_word = dict((i, word) for i, word in enumerate(words_list))

# Generate training data
window_size = 2
training_data = []

for sentence in corpus:
    sent_len = len(sentence)

    for i, word in enumerate(sentence):
        # Convert target word to one-hot encoded vector
        w_target = word2onehot(word=sentence[i], vocab_length=nr_unique_words, word_index=word_index)

        # Cycle through each context window
        w_context = []

        for j in range(i - window_size, i + window_size + 1):
            # Criteria for context word
            # 1. Target word cannot be context word (j != i)
            # 2. Index must be greater than or equal to 0 (j >= 0) - if not, list index out of range
            # 3. Index must be less than or equal to the length of sentence (j <= sent_len -1) - if not, list indext out of range

            if j != i and j<= sent_len-1 and j>= 0:
                # Append the one-hot representation of word to w_context
                w_context.append(word2onehot(word=sentence[j], vocab_length=nr_unique_words, word_index=word_index))
                #print(sentence[i], sentence[j])

        training_data.append([w_target, w_context])


# Train Model

#   Initialise 2 weight matrices (w1 and w2) to facilitate backpropagation
dimension_word_embedding = 10
w1 = np.random.uniform(-1, 1, (nr_unique_words, dimension_word_embedding))
w2 = np.random.uniform(-1, 1, (dimension_word_embedding, nr_unique_words))

epochs = 50
for i in range(epochs):
    # Initialise loss to 0
    loss = 0
    for w_t, w_c in training_data:
        # Forward pass
        # 1. predicted y using softmax (y_pred); 2. matrix of hidden layer (h); 3. output layer before softmax
        y_pred, h, u = forward_pass(target_vector=w_t, weight_matrix1=w1, weight_matrix2=w2)

        #########################################
        # print("Vector for target word:", w_t)	#
        # print("W1-before backprop", w1)	    #
        # print("W2-before backprop", w2)	    #
        #########################################

        # Calculate error
        # 1. For a target word, calculate difference between y_pred and each of the context words
        # 2. Sum up the difference using np.sum to give us the error for this particular target word
        EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
        #########################
        # print("Error", EI)	#
        #########################

        # Backpropagation
        backprop(target_vector=w_t, error=EI, hidden_layer=h, learning_rate=0.01, weight_matrix1=w1, weight_matrix2=w2)
        #########################################
        # print("W1-after backprop", w1)	    #
        # print("W2-after backprop", w2)	    #
        #########################################


# Get vector for word
word = "machine"
vec = word_vec(word=word, word_index=word_index)
print(word, vec)

# Find similar words
vec_sim(word="machine", top_n=3, weight_matrix1=w1, index_word=index_word, word_index=word_index)




