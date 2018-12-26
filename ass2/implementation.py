import tensorflow as tf
import string

'''
BATCH_SIZE:  64, 128
DROP_KEEP_PROB： 0.5， 0.6， 0.7， 0.8
CELL_SIZE：64，128，256
MAX_WORDS_IN_REVIEW： 300，350，400，450

'''

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 200
CELL_SIZE = 64
DROUPOUT_KEEP_PROB = 0.9



EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_CLASSES = 2


stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    # step1: convert all letters into lower cass
    review = review.lower()

    # step2: there maybe exist <br /> in some reviews, we need to strip them.
    br = '<br />'
    review = review.replace(br, ' ')

    # step3: for some common short-hand words, convert them into long-hand
    review = review.replace("won't", " will not")\
            .replace("can't", " can not")\
            .replace("n\'t", " not")\
            .replace("\'re", " are")\
            .replace("\'d", " would")\
            .replace("\'ll", " will")\
            .replace("\'t", " not")\
            .replace("\'ve", " have")\
            .replace("\'m", " am")\
            .replace("\'s", " is")
            
            
            

    # punctuation:  '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # step4: remove normal punctuation and then split the words to a list
    punctuation = string.punctuation
    translator = str.maketrans('', '', punctuation)
    review = review.translate(translator).split()

    # step4: remove the words exist in stop_words
    #        store the words as a string in processed_review
    processed_review = []
    for word in review:
        if word not in stop_words:
            # if processed_review:
            #     processed_review += ' ' + word
            # else:
            #     processed_review = word
            processed_review.append(word)

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    # define input_data with shape: [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE]
    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")

    # define labels with shape: [BATCH_SIZE, NUM_CLASSES]
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES], name="labels")

    # dropout_keep_prob
    dropout_keep_prob = tf.placeholder_with_default(DROUPOUT_KEEP_PROB, shape=(), name="dropout_keep_prob")

    #lstm RNN with Dropout
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)
    init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, dropout_keep_prob)
    outputs, final_state = tf.nn.dynamic_rnn(drop_cell, input_data, dtype=tf.float32)

    # prediction
    weight = tf.Variable(tf.truncated_normal(shape=[CELL_SIZE, NUM_CLASSES], stddev=0.01), name="weight")
    bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="bias")
    logits = tf.matmul(final_state[1], weight) + bias
    preds = tf.nn.softmax(logits)

    # loss
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_mean(xentropy, name="loss")

    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    predicted_labels = tf.argmax(logits, 1)
    real_labels = tf.argmax(labels, 1)
    result = tf.equal(predicted_labels, real_labels)
    Accuracy = tf.reduce_mean(tf.cast(result, tf.float32), name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
