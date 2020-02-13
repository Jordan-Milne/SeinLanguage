import pickle

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open('pipe.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/result', methods=['POST'])
# def result():
#     args = request.form
#     new = pd.DataFrame({
#         'jer_sent': [args.get('jer_sent')],
#         'geo_sent': [args.get('geo_sent')],
#         'kra_sent': [args.get('kra_sent')],
#         'ela_sent': [args.get('ela_sent')],
#         'jer_lines': [args.get('jer_lines')],
#         'geo_lines': [args.get('geo_lines')],
#         'kra_lines': [args.get('kra_lines')],
#         'ela_lines': [args.get('ela_lines')],
#         'location': ['Jerry’s Apartment']
#     })
#     prediction = round(float(pipe.predict(new)[0]),1)
#     return render_template('result.html', prediction=prediction)
#
# @app.route('/generate')
# def generate():
#     vocab = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '<', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#     char2idx = {u:i for i, u in enumerate(vocab)}
#     idx2char = np.array(vocab)
#
#
#
#     def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#       model = tf.keras.Sequential([
#         tf.keras.layers.Embedding(vocab_size, embedding_dim,
#                                   batch_input_shape=[batch_size, None]),
#         tf.keras.layers.LSTM(rnn_units,
#                             return_sequences=True,
#                             stateful=True,
#                             recurrent_initializer='glorot_uniform'),
#         tf.keras.layers.Dense(vocab_size)
#       ])
#       return model
#
#
#
#     # tf.train.latest_checkpoint('ckpt_30.index')
#     checkpoint_dir = './training_checkpoints'
#     model = build_model(86, 256, 1024, batch_size=1)
#     model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#     model.build(tf.TensorShape([1, None]))
#
#     def generate_text(model, start_string):
#       # Evaluation step (generating text using the learned model)
#
#       # Number of characters to generate
#       num_generate = 150
#
#       # Converting start string to numbers
#       input_eval = [char2idx[s] for s in start_string]
#       input_eval = tf.expand_dims(input_eval, 0)
#
#       text_generated = []
#
#       # predictability
#       temperature = 0.1
#
#       # Here batch size == 1
#       model.reset_states()
#
#       for i in range(num_generate):
#           predictions = model(input_eval)
#           # remove the batch dimension
#           predictions = tf.squeeze(predictions, 0)
#
#           # using a categorical distribution to predict the character returned by the model
#           predictions = predictions / temperature
#           predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
#
#           # We pass the predicted character as the next input to the model
#           # along with the previous hidden state
#           input_eval = tf.expand_dims([predicted_id], 0)
#
#           text_generated.append(idx2char[predicted_id])
#
#       return (start_string + ''.join(text_generated))
#
#
#     genlines = str(generate_text(model,"What's the deal with"))
#     return render_template('generate.html', prediction=genlines)

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', debug=True, port=80)
