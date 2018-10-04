import tensorflow as tf
import numpy as np 
import os
import time
import datetime
import text_cnn
import data_processing
from tensorflow.contrib import learn


# Data loading params
tf.flags.DEFINE_string("training_data_file", "./Data/train.txt", "Data source for training")
tf.flags.DEFINE_string("validation_data_file", "./Data/val.txt", "Data source for validating")
tf.flags.DEFINE_string("test_data_file", "./Data/test.txt", "Data source for testing")
# Model Hyperparameters

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

#Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def preprocess():
    # Train + Val data preparation
    print("Loading data...")
    data_train = data_processing.read_data(FLAGS.training_data_file)
    data_val = data_processing.read_data(FLAGS.validation_data_file)
    raw_train_text = data_train[1]
    raw_val_text = data_val[1]
    train_label = data_train[0]
    val_label = data_val[0]
    y_train = []
    y_val = []
    for label in train_label:
        if label == 1:
            pos = [1, 0, 0]
            y_train.append(pos)
        elif label == 2:
            neg = [0, 0, 1]
            y_train.append(neg)
        elif label == 3:
            neu = [0, 1, 0]
            y_train.append(neu)
    for label in val_label:
        if label == 1:
            pos = [1, 0, 0]
            y_val.append(pos)
        elif label == 2:
            neg = [0, 0, 1]
            y_val.append(neg)
        elif label == 3:
            neu = [0, 1, 0]
            y_val.append(neu)


    x_train_text = ""
    x_val_text = ""
    for i in range(len(raw_train_text)):
        clean_sentence = data_processing.sentence_cleaning(raw_train_text[i])
        clean_sentence = " ".join(clean_sentence)
        x_train_text += clean_sentence

    for i in range(len(raw_val_text)):
        clean_sentence = data_processing.sentence_cleaning(raw_val_text[i])
        clean_sentence = " ".join(clean_sentence)
        x_val_text += clean_sentence


    max_sentence_length = 50
    #padORchop(x_text, max_sentence_length)

    # Build vocab
    #vocab = createBoW(x_text)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    x_train = np.array(list(vocab_processor.fit_transform(x_train_text)))
    x_val = np.array(list(vocab_processor.fit_transform(x_val_text)))
    return x_train, y_train, vocab_processor, x_val, y_val


def train(x_train, y_train, vocab_processor, x_val, y_val):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = text_cnn.TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=3,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


            # output directory for models and summaries

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # summaries for loss and accuracy

            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            #train summaries

            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


            #val summaries

            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "summaries", "val")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss: {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, val_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss: {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            batches = data_processing.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs
            )

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    val_step(x_val, y_val, writer=val_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_val, y_val = preprocess()
    train(x_train, y_train, vocab_processor, x_val, y_val)


if __name__ == '__main__':
    tf.app.run()





