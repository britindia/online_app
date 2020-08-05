#! /usr/bin/env python
import sys
import tensorflow as tf
print("tensorfklow version: {}".format(tf.__version__))
import argparse 
import os
import time
from tensorflow.contrib import learn
import numpy as np
import csv



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    
    # Data Parameters
    # raw_input has the review entered by user
    tf.flags.DEFINE_string("frozen_model_filename", "text_cnn3.pb", "Frozen model file to import.")
    tf.flags.DEFINE_string("raw_input", "/tmp/input_reviews.csv", "Raw data for reviews entered Online.")
    tf.flags.DEFINE_string("output_path", "gs://ordinal-reason-282519-aiplatform/text_cnn_training_071320201841", "Current directory.")
    tf.flags.DEFINE_string("raw_text", "", "text entered by the user")
    tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
    
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    
    print("raw_text is: {}".format(FLAGS.raw_text))
    print("raw_input is: {}".format(FLAGS.raw_input))
    print("eval_train is: {}".format(FLAGS.eval_train))
    
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # eval_train will be TRUE if run through flask app. Use below command to run this script standalone  - 
    # ./online_prediction/predict.py --eval_train=False --raw_text="<your review>"
    if FLAGS.eval_train:
        raw_reviews = list(open(FLAGS.raw_input, "r").readlines())
        x_raw = [s.strip() for s in raw_reviews]
        print("x_raw: {}".format(x_raw))
    #     os.remove(os.path.join("/tmp/","input_reviews.csv"))
    else:
        x_raw = [FLAGS.raw_text]
    
    vocab_path = os.path.join(FLAGS.output_path, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    print("x_test: {}".format(x_test))
    
    
    # We use our "load_graph" function
    graph = load_graph(os.path.join(FLAGS.output_path,FLAGS.frozen_model_filename))

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # Get the placeholders from the graph by name

        # prefix/input_x
        # ...
        # prefix/output/predictions
        
    # We access the input and output nodes 
    input_x = graph.get_tensor_by_name('prefix/input_x:0')
    dropout_keep_prob = graph.get_tensor_by_name('prefix/dropout_keep_prob:0')
    # Tensors we want to evaluate
    predictions = graph.get_tensor_by_name('prefix/output/predictions:0')
    
       
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        batch_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
        # below code is to capture text not recognozed by model. We will siplay a sorry message if 
        # model is not able to recognize review
        a=0
        for i in x_test:
            for y in i:
                a += y
        print("a : {}".format(a))
        if a==0:
            batch_predictions=2
            
        # convert numeric sentiment to text
        if batch_predictions == 0:
               sentiment = ["Negative"]
        elif batch_predictions == 1:
               sentiment = ["Positive"]
        else: 
               sentiment = ["Sorry. Model is not able to recognize review. Please enter another review."]
        print("Sentiment : {}".format(sentiment))
        all_predictions = []
        all_predictions = np.concatenate([all_predictions, sentiment])
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path=os.path.join("/tmp/", "online_prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    print("Saving predictions_human_readable to {0}".format(predictions_human_readable))
    with open(out_path, 'w') as f:
       csv.writer(f).writerows(predictions_human_readable)