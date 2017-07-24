'use strict';

const tf = require('tensorflow2');
const graph = tf.graph();
const session = tf.session();
const dataset = require('./dataset');

dataset.load().then((buffers) => {
  const input = graph.constant(buffers[2], tf.dtype.uint8, [10000, 784]);
  const cast = graph.cast(input, tf.dtype.uint8, tf.dtype.float32);
  const xs = session.run(cast);
  console.log('xs converted done');

  const input2 = graph.constant(buffers[3], tf.dtype.uint8, [10000]);
  const onehot = graph.onehot(input2, 10);
  const ys = session.run(onehot);
  console.log('ys converted done');

  graph.load('./train.pb');

  const w = graph.operations.get('var_W/read');
  const assign = graph.state.assign(w, graph.constant(0, tf.dtype.float, [784, 10]));
  session.run(w)

  const op = graph.operations.last();
  const res = session.run(op, {
    x: tf.tensor(xs, tf.dtype.float32, [10000, 784]),
    y_: tf.tensor(ys, tf.dtype.float32, [10000, 10])
  });
  console.log(res);

}).catch((err) => {
  console.error(err && err.stack);
});
