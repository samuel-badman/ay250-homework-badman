### Homework 6 - Supervised Machine Learning - Sam Badman

I have checked this homework in as a jupyter notebook. It is structured as :

* Cell for the grader to input the validation dataset pathname 
* Definition of methods to extract features from image files
* Function to extract all features from a given input image
* Instantiation and evaluation of the random forest classifier using a 10% test
 dataset. Methods to evaluate are : 
 (1) raw percentage of correct identifications
 (2) confusion matrix
 (3) feature importance
* Instantiation of classifier using 100% of the training data, and a function to
call and predict the features for a validation dataset with the pathname 
provided at the top of the notebook, assuming .jpg files only .

I have also checked in a joblib dump of the features and classifications of the training set (~3MB),
and the notebook is configured so that when you run it all, it loads the features from this dump 
instead of re-extracting all the features. This means you can enter the validation file pathname at the
top of the notebook, and then run all cells and the notebook will proceed to train the classifier,
re-run the evaluation cells, and then do the validation, all in less than a minute of run time.

To make the feature rextraction run again, in line [18], change reprocess = False to reprocess = True
The feature extraction runs in parallel using mp.Pool, and on my computer takes around 3 minutes to run.
