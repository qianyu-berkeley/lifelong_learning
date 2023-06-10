# Best Practices

## Neural Network Architecture

- Shape
  - Number of hidden layers
    - Inverted pyramid shape: start out wide and narrow down later
    - Hourglass shape: start out wide, narrow down in the the middle layers, and widen towards the end (common encoder-decoder structure)
    <br>
    <img src="http://files.training.databricks.com/images/hourglass_architecture.png" width="500" height="500">
  - Number of units/neurons in the input and output layers
    - Depends on the input and output your task requires. Example: the MNIST task requires 28 x 28 = 784 input units and 10 output units 
  - Better to increase the number of layers instead of the number of neurons/units per layer
  - Play with the size systematically to compare the performances -- it's a trial-and-error process
  - Two typical approaches:
    - Increase the number of neurons and/or layers until the network starts overfitting
    - Use more layers and neurons than you need, then use early stopping and other regularization techniques to prevent overfitting
  - Borrow ideas from research papers
- Learning Rates:
  - Slanted triangular learning rates: Linearly increase learning rate, followed by linear decrease in learning rate
  <br>
  <img src="https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/slanted_triangular_lr+.png" height="200" width="400">
  - Discriminative Fine Tuning: Different LR per layer
  - Learning rate warmup (start with low LR and change to a higher LR)
  - When the batch size is larger, scale up the learning rate accordingly
  - Use a learning rate finder or scheduler (examples are [here](https://www.avanwyk.com/finding-a-learning-rate-in-tensorflow-2/) and [here](http://d2l.ai/chapter_optimization/lr-scheduler.html#schedulers)) 
- Batch normalization works best when done:
  - after activation functions
  - after dropout layers (if dropout layers are used concurrently)
  - after convolutional or fully connected layers
  
Do as much work as you can on a small data sample


## Regularization Techniques <br>

- Dropout
  - Apply to most type of layers (e.g. fully connected, convolutional, recurrent) and larger networks
  - Set an additional probability for dropping each node (typically set between 0.1 and 0.5)
    - .5 is a good starting place for hidden layers
    - .1 is a good starting place for the input layer
  - Increase the dropout rate for large layers and reduce it for smaller layers
- Early stopping
<br>
 <img src="https://miro.medium.com/max/1247/1*2BvEinjHM4SXt2ge0MOi4w.png" width="500" height="300">
- Reduce learning rate over time
- Weight decay
- Data augmentation
- L1 or L2

Note: No regularization on bias and no weight decay for bias terms <br>

Click to read the [Tensorflow documentation](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#training_procedure) for code examples.


## Convolutional Neural Network

- Layers are typically arranged so that they gradually decrease the spatial resolution of the representations, while increasing the number of channels
- Don't use random initialization. Use [He Initalization](https://arxiv.org/pdf/1502.01852.pdf) to keep the variance of activations constant across every layer, which helps prevent exploding or vanishing gradients
- Label Smoothing: introduces noise for the labels.
<br>
<img src="https://paperswithcode.com/media/methods/image3_1_oTiwmLN.png">
- Max pooling generally performs better than average pooling 
- Image Augmentation:
  - Random crops of rectangular areas in images
  - Random flips
  - Adjust hue, saturation, brightness
  - Note: Use the right kind of augmentation (e.g. don't flip a cat upside down, but satellite image OK)


## Transfer Learning

- Gradual Unfreezing: Unfreeze last layer and train for one epoch and keep unfreezing layers until all layers trained/terminal layer
- Specifically for image use cases:
  - Use a differential learning rate, where the learning rate is determined on a per-layer basis. 
    - Assign lower learning rates to the bottom layers responding to edges and geometrics
    - Assign higher learning rates to the layers responding to more complex features


## Scaling Deep Learning Best Practices

* Use a GPU
* Use Petastorm
* Use Multiple GPUs with Horovod

Click [here](https://databricks.com/blog/2019/08/15/how-not-to-scale-deep-learning-in-6-easy-steps.html) to read the Databricks blog post.


#### References

- [ULMFiT - Language Model Fine-tuning](https://arxiv.org/pdf/1801.06146.pdf)
- [Bag of Tricks for CNN](https://arxiv.org/pdf/1812.01187.pdf)
- [fast.ai](https://forums.fast.ai/t/30-best-practices/12344)
- [Transfer Learning: The Dos and Don'ts](https://medium.com/starschema-blog/transfer-learning-the-dos-and-donts-165729d66625)
- [Dropout vs Batch Normalization](https://link.springer.com/article/10.1007/s11042-019-08453-9)
- [How to Configure the Number of Layers and Nodes](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)
- [Neural networks and Deep Learning by Aurélien Géron](https://www.oreilly.com/library/view/neural-networks-and/9781492037354/ch01.html)