import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict

training_data = pd.read_csv("Corona_train.csv")
validation_data = pd.read_csv("Corona_validation.csv")

Y_train = training_data["Sentiment"]
Y_validation = validation_data["Sentiment"]
# Feature Extraction (Unigrams and Bigrams)
def extract_ngrams(text, n):
    words = text.split()
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return ngrams


# Generate unigram and bigram features
training_data['unigrams'] = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
training_data['bigrams'] = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2))

validation_data['unigrams'] = validation_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
validation_data['bigrams'] = validation_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2))

# Build Vocabulary
vocab = set()

for el in training_data["unigrams"]:
    vocab.update(el)
for el in training_data["bigrams"]:
    vocab.update(el)

# Model Training
class NaiveBayesClassifier:
    def __init__(self, vocab):
        self.vocab = vocab
        self.class_prob = defaultdict(float)
        self.class_count = defaultdict(float)
        self.word_prob = defaultdict(lambda: defaultdict(float))
        self.word_count = defaultdict(lambda: defaultdict(float))

    def train(self, train_data, bigram = False, percent_100 = False, trigram = False):
        # percent_100 is a parameter only for 100% target domain in part (f)
        # I have this parameter to handle a discrepency in the data .
        total_documents = len(train_data)
        V = len(self.vocab)
        for _, row in train_data.iterrows():
            if percent_100:
                label = row['Sentimwnt']
            else:
                label = row['Sentiment']
            self.class_prob[label] += 1
            self.class_count[label] += 1
            for unigrams in row['unigrams']:
                self.word_prob[label][unigrams] += 1
                self.word_count[label][unigrams] += 1
            if bigram:
                for bigrams in row['bigrams']:
                    self.word_prob[label][bigrams] += 1
                    self.word_count[label][bigrams] += 1
            if trigram:
                for trigrams in row['trigrams']:
                    self.word_prob[label][trigrams] += 1
                    self.word_count[label][trigrams] += 1

        self.class_prob["Positive"] /= total_documents
        self.class_prob["Neutral"] /= total_documents
        self.class_prob["Negative"] /= total_documents

        for label in self.word_prob:
            total_words = sum(self.word_prob[label].values())
            for ngram in self.word_prob[label]:
                self.word_prob[label][ngram] = (self.word_prob[label][ngram] + 0.02)/(total_words + 0.02*V)

    def predict(self, test_data, bigram = False, trigram = False):
        self.predictions = []
        V = len(self.vocab)
        for _, row in test_data.iterrows():
            scores = defaultdict(float)
            for label in self.class_prob:
                scores[label] = np.log(self.class_prob[label])

                for unigrams in row['unigrams']:
                    scores[label] += np.log(self.word_prob[label].get(unigrams, 1/V))
                if bigram:
                    for bigrams in row['bigrams']:
                        scores[label] += np.log(self.word_prob[label].get(bigrams, 1/V))
                if trigram:
                    for trigrams in row['trigrams']:
                        scores[label] += np.log(self.word_prob[label].get(trigrams, 1/V))

            self.predictions.append(max(scores, key=scores.get))
        return self.predictions

    def wordcloud(self, class_label):
        word_freq_dict = {word : self.word_count[class_label][word] for word in self.vocab}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def confusion_matrix(self, test_data, predicted_data):
        # The row 0 in confusion matrix correspond to actual class == Positive.
        # The row 1 in confusion matrix correspond to actual class == Neutral.
        # The row 2 in confusion matrix correspond to actual class == Negative.
        # The column 0 in confusion matrix correspond to predicted class == Positive.
        # The column 1 in confusion matrix correspond to predicted class == Neutral.
        # The column 2 in confusion matrix correspond to predicted class == Negative.

        confusion_mat = np.zeros((3, 3))
        label_to_index = {"Positive":0, "Neutral":1, "Negative":2}
        for i in range(len(test_data)):
            confusion_mat[label_to_index[test_data[i]]][label_to_index[predicted_data[i]]] += 1

        ##### Now draw the confusion matrix #####
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_mat, cmap='Blues')

        # Add labels for the axes
        ax.set_xticks(np.arange(confusion_mat.shape[1]))
        ax.set_yticks(np.arange(confusion_mat.shape[0]))
        ax.set_xticklabels(['Positive class', 'Neutral class', 'Negative class'])
        ax.set_yticklabels(['Positive class', 'Neutral class', 'Negative class'])

        # Rotate the x-axis labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                text = ax.text(j, i, str(confusion_mat[i, j]), 
                               ha="center", va="center", color="black")
 
        # Display the colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")

        # Set labels and title
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")

        plt.show()

################# (a) Part-(i) ###################

# Naive Bayes model for it.
classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data)

# Model Evaluation
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
print("Accuracy on validation set:", accuracy * 100)

Y_pred_train = classifier.predict(training_data)

n1 = len(Y_pred_train)
accuracy_train = 0
for i in range(n1):
    if Y_pred_train[i] == Y_train[i]:
        accuracy_train += 1/n1

print("Accuracy training set:", accuracy_train * 100)

################## (a) part- (ii) ###################

classifier.wordcloud("Positive")
classifier.wordcloud("Neutral")
classifier.wordcloud("Negative")

################## (b) part-(i) #####################

## Calculating accuracy for validation set
choices = ["Positive", "Neutral", "Negative"]
Y_val_rnd = np.random.choice(choices, size = len(Y_validation))
n1 = len(Y_validation)
accuracy_val_random = 0
for j in range(len(Y_validation)):
    if Y_val_rnd[j] == Y_validation[j]:
        accuracy_val_random += 1/n1
print("Accuracy for random model on validation set:", accuracy_val_random * 100)

## Calculating accuracy for training set
choices = ["Positive", "Neutral", "Negative"]
Y_train_rnd = np.random.choice(choices, size = len(Y_train))
n1 = len(Y_train)
accuracy_train_random = 0
for j in range(len(Y_train)):
    if Y_train_rnd[j] == Y_train[j]:
        accuracy_train_random += 1/n1
print("Accuracy for random model on training set:", accuracy_train_random * 100)

################ (b) Part-(ii) ################

## Calculating accuracy for validation set
Y_val_pos = np.array(["Positive"] * len(Y_validation))
n1 = len(Y_validation)
accuracy_val_positive = 0
for j in range(len(Y_validation)):
    if Y_val_pos[j] == Y_validation[j]:
        accuracy_val_positive += 1/n1
print("Accuracy for positive model on validation set:", accuracy_val_positive * 100)

## Calculating accuracy for training set
Y_train_pos = np.array(["Positive"] * len(Y_train))
n1 = len(Y_train)
accuracy_train_positive = 0
for j in range(len(Y_train)):
    if Y_train_pos[j] == Y_train[j]:
        accuracy_train_positive += 1/n1
print("Accuracy for positive model on training set:", accuracy_train_positive * 100)

############### (b) Part-(iii) ################

print("Accuracy amount (in %) by which my model is better than the random one:", (accuracy - accuracy_val_random) * 100)
print("Accuracy amount (in %) by which my model is better than the positive one:", (accuracy - accuracy_val_positive) * 100)

############### (c) Part-(i) ##################

classifier.confusion_matrix(Y_validation, Y_pred_val)
classifier.confusion_matrix(Y_train, Y_pred_train)
classifier.confusion_matrix(Y_validation, Y_val_rnd)
classifier.confusion_matrix(Y_train, Y_train_rnd)
classifier.confusion_matrix(Y_validation, Y_val_pos)
classifier.confusion_matrix(Y_train, Y_train_pos)

################ (d) part-(i) ################

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter stemmer and stopwords set
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text1 = []
    for word in text:
        if word.lower() not in stop_words and word not in string.punctuation:
            text1.append(word)

    # Lowercase the words
    text2 = [word.lower() for word in text1]

    # Stemming
    stemmer = PorterStemmer()
    text3 = [stemmer.stem(word) for word in text2]

    # Join the words back into a string
    preprocessed_text = ' '.join(text3)
    #print(preprocessed_text)
    return text3

# Preprocess training data
training_data["unigrams"] = training_data["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

vocab = set()

for el in training_data["unigrams"]:
    vocab.update(el)
for el in training_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data)
classifier.wordcloud("Positive")
classifier.wordcloud("Neutral")
classifier.wordcloud("Negative")
# Model Evaluation
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
print("Accuracy on validation set:", accuracy * 100)

Y_pred_train = classifier.predict(training_data)

n1 = len(Y_pred_train)
accuracy_train = 0
for i in range(n1):
    if Y_pred_train[i] == Y_train[i]:
        accuracy_train += 1/n1

print("Accuracy training set:", accuracy_train * 100)


################# Q1.(e) Part-(i) ###################

# Train the Naive Bayes model

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data, True)
training_data["bigrams"] = training_data["bigrams"].apply(lambda text: preprocess_text(text))

validation_data["bigrams"] = validation_data["bigrams"].apply(lambda text:preprocess_text(text))

# Model Evaluation
Y_pred = classifier.predict(validation_data, True)

# Calculate Accuracy
n1 = len(Y_pred)
accuracy = 0
for i in range(n1):
    if Y_pred[i] == Y_validation[i]:
        accuracy += 1/n1
print("Accuracy on validation set using bigrams:", accuracy * 100)

################# Q1.(e) Part-(ii) ###################
"""
training_data['trigrams'] = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 3))

validation_data['trigrams'] = validation_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 3))

training_data["trigrams"] = training_data["trigrams"].apply(lambda text: preprocess_text(text))

validation_data["trigrams"] = validation_data["trigrams"].apply(lambda text:preprocess_text(text))

for el in training_data["trigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data, True, False, True)
Y_pred = classifier.predict(validation_data, True, True)

# Calculate Accuracy
n1 = len(Y_pred)
accuracy = 0
for i in range(n1):
    if Y_pred[i] == Y_validation[i]:
        accuracy += 1/n1
print("Accuracy on validation set using trigrams:", accuracy * 100)
"""

################# Q1.(f) Part-(i) ###################

# list to store the accuracy values when trained on source and a proprotion of target domain
accuracy_st = []    
# list to store the accuracy values when trained only on a proprotion of target domain
accuracy_t = []

### Model for 1 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_1.csv")
validation_data = pd.read_csv("Twitter_validation.csv")
validation_data["CoronaTweet"] = validation_data["Tweet"]
c1 = validation_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))
c2 = validation_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2))
validation_data["unigrams"] = c1
validation_data["bigrams"] = c2

merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))
#print(training_data1)

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 1% target domain:", accuracy * 100)

# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)
print("Accuracy on validation set by training on 1% target domain:", accuracy * 100)

### Model for 2 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_2.csv")
merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el) 
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 2% target domain:", accuracy * 100)

# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)
print("Accuracy on validation set by training on 2% target domain:", accuracy * 100)

### Model for 5 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_5.csv")

merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 5% target domain:", accuracy * 100)

# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)
print("Accuracy on validation set by training on 5% target domain:", accuracy * 100)

### Model for 10 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_10.csv")


merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 10% target domain:", accuracy * 100)


# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)
print("Accuracy on validation set by training on 10% target domain:", accuracy * 100)

### Model for 25 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_25.csv")


merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 25% target domain:", accuracy * 100)


# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)        
print("Accuracy on validation set by training on 25% target domain:", accuracy * 100)

### Model for 50 % target training dataset ###

training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_50.csv")


merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentiment"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 50% target domain:", accuracy * 100)

# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy)
print("Accuracy on validation set by training on 50% target domain:", accuracy * 100)

### Model for 100 % target training dataset ###
############ Note - The dataset contains a discrepancy which is Sentiment is written
############ as Sentimwnt. So, I have adjusted my code to handle it.
 
training_data = pd.read_csv("Corona_train.csv")
target_data = pd.read_csv("Twitter_train_100.csv")

merged_column1 = pd.concat([training_data["Sentiment"], target_data["Sentimwnt"]], ignore_index= True)
merged_column2 = pd.concat([training_data["CoronaTweet"], target_data["Tweet"]], ignore_index= True)
c1 = training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 1))
c1 = c1.apply(lambda text: preprocess_text(text))

c2 = target_data["Tweet"].apply(lambda x: extract_ngrams(x, 1))
c2 = c2.apply(lambda text: preprocess_text(text))
target_data["unigrams"] = c2
target_data["bigrams"] = target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))
merged_column3 = pd.concat([c1, c2], ignore_index= True)
merged_column4 = pd.concat([training_data["CoronaTweet"].apply(lambda x: extract_ngrams(x, 2)), target_data["Tweet"].apply(lambda x:extract_ngrams(x, 2))], ignore_index= True)
training_data1 = pd.DataFrame({"Sentiment": merged_column1, "CoronaTweet": merged_column2, "unigrams": merged_column3, "bigrams":merged_column4})
training_data1["unigrams"] = training_data1["unigrams"].apply(lambda text: preprocess_text(text))

validation_data["unigrams"] = validation_data["unigrams"].apply(lambda text:preprocess_text(text))

# Build Vocabulary
vocab = set()

for el in training_data1["unigrams"]:
    vocab.update(el)
for el in training_data1["bigrams"]:
    vocab.update(el)

Y_train = training_data1["Sentiment"]
Y_validation = validation_data["Sentiment"]

classifier = NaiveBayesClassifier(vocab)
classifier.train(training_data1)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_st.append(accuracy)
print("Accuracy on validation set by training on source domain and 100% target domain:", accuracy * 100)


# Build Vocabulary
vocab = set()

for el in target_data["unigrams"]:
    vocab.update(el)
for el in target_data["bigrams"]:
    vocab.update(el)

classifier = NaiveBayesClassifier(vocab)
classifier.train(target_data, False, True)
Y_pred_val = classifier.predict(validation_data)

# Calculate Accuracy
n1 = len(Y_pred_val)
accuracy = 0
for i in range(n1):
    if Y_pred_val[i] == Y_validation[i]:
        accuracy += 1/n1
accuracy_t.append(accuracy) 
print("Accuracy on validation set by training on 100% target domain:", accuracy * 100)

####### Q1.(f) Part - (iii) #######

# The blue line graph represents the validation set accuracy measured by training on
# source domain and a proprotion of target domain
# The red line graph represents the validation set accuracy measured by training 
# only on a proportion of target domain.

accuracy_st = np.array(accuracy_st)
accuracy_t = np.array(accuracy_t)
target_train_proportions = np.array([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
plt.plot(target_train_proportions, accuracy_st, label='Source + Target Data', color = "blue")
plt.plot(target_train_proportions, accuracy_t, label='Target Data Only', color = "red")
plt.xlabel('Proportion of Target Training Data')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
