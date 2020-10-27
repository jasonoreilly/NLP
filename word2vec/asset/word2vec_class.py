import numpy as np

from collections import defaultdict


def softmax(target_vector):
    e_x = np.exp(target_vector - np.max(target_vector))
    softmaxed = e_x / e_x.sum(axis=0)

    return softmaxed


class word2vec:

    def __init__(self, window_size, number_training_epochs, learning_rate, dimensions_word_embeddings, input_text):
        self.window = window_size
        self.epochs = number_training_epochs
        self.lr = learning_rate
        self.n = dimensions_word_embeddings
        self.text = input_text
        self.corpus = [[word.lower() for word in input_text.split()]]

    def derive_parameters_from_input_text(self):
        # Find unique word counts using dictonary
        word_counts = defaultdict(int)
        for row in self.corpus:
            for word in row:
                word_counts[word] += 1
        # print(word_counts)

        # Initialise vocab size
        self.nr_unique_words = len(word_counts.keys())
        # print(self.nr_unique_words)

        # Initialise lookup dictionary (unique words only)
        self.words_list = list(word_counts.keys())
        # print(self.words_list)

        # Generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # print(self.word_index)

        # Generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        # print(self.index_word)

    def generate_training_data(self):
        training_data = []

        for sentence in self.corpus:
            sent_len = len(sentence)

            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])
                print(w_target)

                # Cycle through each context window
                w_context = []

                # Note: window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window + 1):
                    # Criteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range

                    if j != i and j <= sent_len - 1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])

                # print(training_data)
                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        """
        creates a vector of dimension 1xvocab_size with all zeros except for one position (1), corresponding to the word_index position

        :param word:
        :return:
        """
        # Initialise a blank vector
        word_vec = [0 for i in range(0, self.nr_unique_words)]                      # does not work! initialised wronlgy (index problem @loss step) np.zeros(self.nr_unique_words)

        # Get ID of word from word_index
        word_id = self.word_index[word]

        # Change the value from 0 to 1 at the target word's position
        word_vec[word_id] = 1

        return word_vec

    def train_model(self, training_data):
        self.weight_matrix1 = np.random.uniform(-1, 1, (self.nr_unique_words, self.n))       # 9x10 matrix (since we will calculate 1x9 dot 9x10 = 1x10)
        self.weight_matrix2 = np.random.uniform(-1, 1, (self.n, self.nr_unique_words))       # 10x9 matrix (since we wil calculate 1x10 dot 10x9 = 1x9)
        # print(self.weight_matrix1.shape, self.weight_matrix2.shape)

        # Cycle through each epoch
        for i in range(self.epochs):
            # Initialise loss to 0
            self.loss = 0
            for w_t, w_c in training_data:
                # Forward pass
                # Outputs: 1. predicted y using softmax (y_pred); 2. matrix of hidden layer (h); 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(target_vector=w_t)

                # Calculate error
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the difference using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                # print("Error", EI)

                # Backpropagation
                self.backprop(target_vector=w_t, error=EI, hidden_layer=h)

                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1:  negative of the sum for all the elements in the output layer (before softmax)
                # Part 2: number of the context words and multiplies the log of sum for all elements (after exponential) in the output layer
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, target_vector):
        # x is one-hot encoded vector, shape 1*(nr_unique_words) (1x9)
        # Run through first matrix (w1) to get hidden layer - 1x9 dot 9x10 returns 1x10
        h = np.dot(target_vector, self.weight_matrix1)
        # Dot product hidden layer with second matrix (w2) - 1x10 dot 10x9 returns 1x9 - one for each unique_word
        u = np.dot(h, self.weight_matrix2)
        # Run 1x9 through softmax to force each element to range of [0, 1] (probabilites) - 1x9
        y_c = softmax(u)
        # print(h.shape, u.shape, y_c.shape)

        return y_c, h, u

    def backprop(self, error, hidden_layer, target_vector):
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backward, we need to take the derivative of E with respect to w2
        # h - shape 10x1, e - shape 9x1, d1_dw2 - shape 10x9
        # x shape 9x1, w2 - shape 10x9, e.T - shape 9x1
        dl_dw2 = np.outer(hidden_layer, error)
        dl_dw1 = np.outer(target_vector, np.dot(self.weight_matrix2, error.T))

        #########################################
        # print('Delta for w2', dl_dw2)			#
        # print('Hidden layer', h)				#
        # print('np.dot', np.dot(w2, e.T))	    #
        # print('Delta for w1', dl_dw1)			#
        #########################################

        # Update weights
        self.weight_matrix1 = self.weight_matrix1 - (self.lr * dl_dw1)
        self.weight_matrix2 = self.weight_matrix2 - (self.lr * dl_dw2)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.weight_matrix1[w_index]

        return v_w

    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.nr_unique_words):
            # Find similarity score for each word in vocab
            v_w2 = self.weight_matrix1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)


model = word2vec(input_text="natural language processing and machine learning is fun and exciting",
                 window_size=2, number_training_epochs=100, learning_rate=0.01, dimensions_word_embeddings=10)

model.derive_parameters_from_input_text()

train_data = model.generate_training_data()

model.train_model(training_data=train_data)

# Get vector for word
word = "natural"
vec = model.word_vec(word)
print(word, vec)

# Find similar words
model.vec_sim("natural", 3)


