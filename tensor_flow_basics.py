#
# Me Learning Machine Learning
# Part 2: Python and TensorFlow
#
# I am basically trying things out, as I follow the Getting Started documents
# in TensorFlow's official web site (https://www.tensorflow.org/get_started/).
#
# Again, expect lots of noob-style "this line does x, y and z" comments.
#
# Leandro Motta Barros
#

import tensorflow as tf # The canonical way to import TensorFlow

# This helps to see where my stuff starts on the console :-)
print("---------------------------------")

# This is the Session. It is used to execute stuff. "Operations" would be the
# more technical name for "stuff", I guess. Computational Graphs seem to be one
# particularly important kind of Operation.
sess = tf.Session()


# ------------------------------------------------------------------------------
#  Toying with graphs
# ------------------------------------------------------------------------------


# Speaking of Computational Graphs, they are what we use to represent our
# models. Let's build one. First, create some constants. We can state the
# desired type explicitly, as in the first case, but the type is also inferred
# automaticaly from the kind of literal used, as in the other cases. All these
# constants are of type `tf.float32`.
const100 = tf.constant(100.0, dtype=tf.float32)
const250 = tf.constant(250.0)
const2 = tf.constant(2.0)

# Now, create an operation between these nodes
adderNode = tf.add(const100, const250)

# Run the operation
print("adderNode -->", sess.run([adderNode]))

# There is syntatic sugar for creating nodes that do simple operations.
altAdderNode = const100 + const100 + const250
subNode = const250 - const100
powerNode = const250 ** const2

print("altAdderNode -->", sess.run([altAdderNode]))
print("subNode -->", sess.run([subNode]))
print("powerNode -->", sess.run([powerNode]))

# Placeholders allow to pass input data to our graphs. In the words used in the
# tutorial, "a placeholder is a promise to provide a value later".
phA = tf.placeholder(tf.float32)
phB = tf.placeholder(tf.float32)

phOpNode = (phA + phB + const100) / const250

# A dictionary is used to pass the desired values when running the node. Note
# that TensorFlow works with tensors, not just with scalar values.
print("phOpNode [1] -->", sess.run(phOpNode, {phA: 100, phB: 111}))
print("phOpNode [2] -->", sess.run(phOpNode, {phA: 200, phB: 500}))
print("phOpNode [3] -->", sess.run(phOpNode, {phA: [100, 200], phB: [111, 500]}))
print("phOpNode [4] -->", sess.run(phOpNode, {
    phA: [[100, 200], [333, 171]],
    phB: [[111, 500], [171, 333]]}))

# Combining graphs into more complex graphs also works just as one would expect.
# Even that hardcoded `400` constant does what it should.
compositeNode = subNode + phOpNode * 400 / adderNode
print("compositeNode --> ", sess.run(compositeNode, {phA: 100, phB: 111}))


# ------------------------------------------------------------------------------
#  A linear model
# ------------------------------------------------------------------------------

# So far, the models we created aren't very useful. Machine Learning is about
# modeling and optimizing he models, right? My models so far were just a bunch
# of random operations operations running on random inputs (via Placeholders). A
# real ML model needs parameters to be optimized during training, and parameters
# are represented in TensorFlow by Variables.

# So let's build a more reasonable model, and one that is parameterized by
# two Variables: a linear model (y = a*x + b).

# The first variable, `a`, is the be the line slope. We instantiate passing an
# initial value (`[0.3]`, which is a tensor) and an explicit type.
a = tf.Variable([0.3], dtype=tf.float32)

# The second variable, `b`, is the line y-intercept. Here, we let the type be
# inferred (it will be a `tf.float32` also).
b = tf.Variable([-.3])

# We also need the input variable `x`. Nothing new here:
x = tf.placeholder(tf.float32)

# Now we can finally define our linear model:
linearModel = a * x + b

# For some reason (I conjecture about this reason below), the initial values of
# Variables are not automaticaly assigned. We need to create and run a subgraph
# that does this initialization. Fortunately, there is a function to create this
# subgraph:
initGraph = tf.global_variables_initializer()
sess.run(initGraph)

# At this point we can run our model, just as we did before:
print("linearModel [1] -->", sess.run(linearModel, {x: 2}))
print("linearModel [2] -->", sess.run(linearModel, {x: [1, 2, 3, 4]}))


# ------------------------------------------------------------------------------
#  Defining a loss function (AKA cost function)
# ------------------------------------------------------------------------------

# If we intend to have any (supervised) training happening, we need a loss
# function. Let's define it as being the sum of square errors. Now, I am not
# sure I fully understand the TensorFlow-specific things here, but I'd say the
# loss function is just another graph -- one that represents the, well, loss
# function (instead of the model).
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linearModel - y))

# Now we can define our training data and check how well our model is doing:
trainingData = {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}
print("loss [1] -->", sess.run(loss, trainingData))

# We can also manually change the model parameters (Variables) and re-check how
# it is doing. The idea is simple enough, but expressing it in TensorFlow is a
# bit funny.
#
# BTW, at this point I had a mini-epiphany: TensorFlow is a
# graph-processing machine, with graphs representing operations within a session
# (this seems to be true at least to a certain extent, though I cannot tell yet
# what extent is this). Our model is a graph. Our loss function is a graph. The
# initialization of Variables needs to be executed in a graph because the
# variables must be initialized in the "TensorFlow side", not in the "Python
# side".
#
# And here are we, trying to change the value of Variables. We cannot simply
# assign to these variables, because the reside in the "TensorFlow side". We
# need to create graphs that do the assignments, then execute the graphs:
assignA = tf.assign(a, [-0.9])
assignB = tf.assign(b, [0.9])
sess.run([assignA, assignB]) # run both graphs at once
print("loss [2] -->", sess.run(loss, trainingData))

# ------------------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------------------

# Changing Variables manually isn't real Machine Learning. TensorFlow provides
# optimizers that will do this work for us. In this case, we'll use a plain
# gradient descent optimizer. Again, the optimization process consists of
# creating a graph and running it. That's what TensorFlow seems to do all the
# time: run graphs.

# Create am optimizer with a certain learning rate
learningRate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learningRate)

# Create our training graph: it will run the optimizer so that it minimizes
# `loss`.
train = optimizer.minimize(loss)

# Just avoid being called a cheater, reset the Variables to their initial
# values.
sess.run(initGraph)

# Run the `train` graph. Notice that we need to implement the loop manually
# (here, for a fixed number of iterations):
for i in range(1000):
    sess.run(train, trainingData)

# How do we get the optimal values? Well, running a graph, of course! Which
# graph? The same graph that represents the Variable:
print("Optimal a -->", sess.run(a))
print("Optimal b -->", sess.run(b))
print("Both at once -->", sess.run([a, b]))

# Check the loss. Hopefully, will be pretty close to zero.
print("loss [final] -->", sess.run(loss, trainingData))
