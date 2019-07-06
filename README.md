### Wine Review: Multi-Classification of Wine Varieties by Deep Learning
Here we introduce a neural network model for multi-classification of wine varieties from Wine Reviews datasets. The model consists of a bidirectional LSTM encoder for representatively embedding the review  and a prediction network for classification. Using only the learned embeddings as input, the model results in accuracy as high as 77% for 10-classes classification and well-balanced precision and recall. By concatenating both the learned embeddings and external features as input, the model accuracy has been improved to >82%.

##Introduction

It is of great satisfaction for wine lover if he/she could identify its variety (and even winery and manufacturing year) by having a simple taste. Such practice could be possible for those who taste wines frequently from a wide range of varieties (as well as from a wide range of wineries) like professional wine tasters. Artificial intelligent has shown dramatic progress in the past decade that has transformed practice in a variety of fields such as drug discovery, speech recognition, image processing, etc. Thus, it is natural to ask how could it perform to identify a wine variety (and even manufacturing winery) just based on tasters' reviews?

Recurrent neural networks (RNNs) have shown great success in many NLP tasks. In particular, long short term memory networks (LSTM), capable of learning long term dependencies and avoiding gradient vanishing and exploding, has been widely adopted to recognize context-sensitive languages.  The reviews, not only for wine but also for others like restaurants, hotels, are typically sentimental to some extent. Thus, we expect that LSTM could be a great tool for 'understanding' the underlying wine reviews. 

##Preprocess

![Description_length_distribution](https://user-images.githubusercontent.com/34787111/60202040-a4918b80-97fe-11e9-8b3e-e5e1e81422c1.png)

Figure 1. Statistical distribution of wine review length.

Given that the goal is to predict wine variety from the review (column description in the dataset), converting textual description to ML-learnable embedding is the primary task for data preprocessing. However, the beauty of deep learning model, unlike bag of word approach,  is that it is not required to preprocess the text. The text will be directly fed into model for learning. As comparison, we did preprocess the reviews for tfidf embedding. Figure 1 shows the review length distribution labeled by wine varieties. Interestingly, almost all length distribution from all varieties are normally distributed.

We did extract / preprocess other features from the dataset, such as manufactured year, country, province, taster name, points, and price. As usual, we imputer missing values and then preprocess them for learning. Please refer to the code for details.

##Model

![NN_architecture](https://user-images.githubusercontent.com/34787111/60202610-c93a3300-97ff-11e9-80f1-a19dbaeec831.png)

Figure 2. Neural network model architecture consisting of a textural representative embedding and a prediction network.



![variety_cm](https://user-images.githubusercontent.com/34787111/60761356-69ac0680-9ffb-11e9-8fd0-c437b58ba431.png)

Figure 3. Confusion matrix of the predicted results for the test data. (a): model with input of the textual embeddings only, accuracy 77%; (b): model with input of both the textual embeddings and other features, accuracy 82%.


##Conclusion





