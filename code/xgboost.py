
import utils
import random
import numpy as np
from xgboost import XGBClassifier
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer

# Configuration Variables
FREQ_DIST_FILE = '../train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = '../train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = '../train-processed.csv'
TEST_PROCESSED_FILE = '../test-processed.csv'
TRAIN = True
UNIGRAM_SIZE = 15000
VOCAB_SIZE = UNIGRAM_SIZE
USE_BIGRAMS = True
if USE_BIGRAMS:
    BIGRAM_SIZE = 10000
    VOCAB_SIZE += BIGRAM_SIZE  # Combined Unigram and Bigram Size
FEAT_TYPE = 'frequency'

# Helper function to extract unigrams and bigrams
def get_feature_vector(tweet):
    uni_feature_vector = []
    bi_feature_vector = []
    words = tweet.split()
    for i in range(len(words) - 1):  # xrange replaced with range
        word = words[i]
        next_word = words[i + 1]
        if unigrams.get(word):
            uni_feature_vector.append(word)
        if USE_BIGRAMS and bigrams.get((word, next_word)):
            bi_feature_vector.append((word, next_word))
    if len(words) >= 1 and unigrams.get(words[-1]):
        uni_feature_vector.append(words[-1])
    return uni_feature_vector, bi_feature_vector

# Function to extract features from the tweets dataset
def extract_features(tweets, batch_size=500, test_file=True, feat_type='presence'):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))
    for i in range(num_batches):  # xrange replaced with range
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = lil_matrix((batch_size, VOCAB_SIZE))
        labels = np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words, tweet_bigrams = tweet[1]
            else:
                tweet_words, tweet_bigrams = tweet[2]
                labels[j] = tweet[1]

            if feat_type == 'presence':
                tweet_words = set(tweet_words)
                tweet_bigrams = set(tweet_bigrams)
            for word in tweet_words:
                idx = unigrams.get(word)
                if idx:
                    features[j, idx] += 1
            if USE_BIGRAMS:
                for bigram in tweet_bigrams:
                    idx = bigrams.get(bigram)
                    if idx:
                        features[j, UNIGRAM_SIZE + idx] += 1
        yield features, labels

# Apply TF-IDF transformation to features
def apply_tf_idf(X):
    transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    return transformer.fit_transform(X)  # Direct transformation after fitting

# Process and return tweets with features
def process_tweets(csv_file, test_file=True):
    tweets = []
    print('Generating feature vectors')  # Print format changed
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            utils.write_status(i + 1, total)
    print('\nProcessing complete')
    return tweets

if __name__ == '__main__':
    np.random.seed(1337)
    unigrams = utils.top_n_words(FREQ_DIST_FILE, UNIGRAM_SIZE)
    if USE_BIGRAMS:
        bigrams = utils.top_n_bigrams(BI_FREQ_DIST_FILE, BIGRAM_SIZE)
    
    tweets = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)

    if TRAIN:
        train_tweets, val_tweets = utils.split_data(tweets)
    else:
        random.shuffle(tweets)
        train_tweets = tweets

    del tweets  # Free up memory

    print('Extracting features & training batches')
    clf = XGBClassifier(max_depth=25, verbosity=1, n_estimators=400)  # Changed 'silent' to 'verbosity'
    
    batch_size = len(train_tweets)
    i = 1
    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    
    for training_set_X, training_set_y in extract_features(train_tweets, test_file=False, feat_type=FEAT_TYPE, batch_size=batch_size):
        utils.write_status(i, n_train_batches)
        i += 1
        if FEAT_TYPE == 'frequency':
            training_set_X = apply_tf_idf(training_set_X)
        clf.fit(training_set_X, training_set_y)

    print('\nTesting model performance')
    if TRAIN:
        correct, total = 0, len(val_tweets)
        i = 1
        batch_size = len(val_tweets)
        n_val_batches = int(np.ceil(len(val_tweets) / float(batch_size)))
        
        for val_set_X, val_set_y in extract_features(val_tweets, test_file=False, feat_type=FEAT_TYPE, batch_size=batch_size):
            if FEAT_TYPE == 'frequency':
                val_set_X = apply_tf_idf(val_set_X)
            prediction = clf.predict(val_set_X)
            correct += np.sum(prediction == val_set_y)
            utils.write_status(i, n_val_batches)
            i += 1
        
        accuracy = correct * 100. / total
        print(f'\nCorrect: {correct}/{total} = {accuracy:.4f}%')
    else:
        del train_tweets
        test_tweets = process_tweets(TEST_PROCESSED_FILE, test_file=True)
        predictions = np.array([])
        print('Predicting test set batches')
        i = 1
        n_test_batches = int(np.ceil(len(test_tweets) / float(batch_size)))
        
        for test_set_X, _ in extract_features(test_tweets, test_file=True, feat_type=FEAT_TYPE):
            if FEAT_TYPE == 'frequency':
                test_set_X = apply_tf_idf(test_set_X)
            prediction = clf.predict(test_set_X)
            predictions = np.concatenate((predictions, prediction))
            utils.write_status(i, n_test_batches)
            i += 1
        
        predictions = [(str(j), int(predictions[j])) for j in range(len(test_tweets))]
        utils.save_results_to_csv(predictions, 'xgboost_predictions.csv')
        print('\nResults saved to xgboost_predictions.csv')
