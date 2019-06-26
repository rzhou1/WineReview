### Wine Review: Multi-Classification of Wine Varieties by Deep Learning
Here we introduce a neural network model for multi-classification of wine varieties from Wine Reviews datasets. The model consists of a bidirectional LSTM encoder for textual representative embeddings (here the wine review) and a prediction network for classification. Using only the learned embeddings as input, the model results in accuracy as high as 77% for 10-classes classification and well-balanced precision and recall. By concatenating both the learned embeddings and external features as input, the model accuracy has been improved to 82%.

##Introduction

It is of great satisfaction for wine lover if he/she could identify its variety (and even winery and manufacturing year) by having a simple taste. Such practice could be possible for those who taste wines frequently from a wide range of varieties (as well as from a wide range of wineries) like professional wine tasters. 

##Data preprocess
![Description_length_distribution](https://user-images.githubusercontent.com/34787111/60202040-a4918b80-97fe-11e9-8b3e-e5e1e81422c1.png)

Given that the goal is to predict wine variety from the review (column description in the dataset), converting textual description to ML-learnable embedding is the primary task for data preprocessing. 

![Price_dist](https://user-images.githubusercontent.com/34787111/60202059-ac513000-97fe-11e9-98ae-9b0e400f9a0d.png)

##Model
[NN_architecture.pptx](https://github.com/rzhou1/WineReview/files/3331293/NN_architecture.pptx)


![variety_NN_others51](https://user-images.githubusercontent.com/34787111/60202086-b5420180-97fe-11e9-85bb-c85696cf5e3b.png)

![variety_NN_desc_only_77](https://user-images.githubusercontent.com/34787111/60202097-bbd07900-97fe-11e9-9d11-876a2600187e.png)

![variety_NN_std_82](https://user-images.githubusercontent.com/34787111/60202114-c12dc380-97fe-11e9-8427-257b52b94a42.png)

##Conclusion





