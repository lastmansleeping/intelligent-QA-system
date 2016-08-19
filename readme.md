#Intelligent Question Answering Engine
* Trained a logistic regression classifier, which functioned as an answer sentence retrieval system, over the WikiQA dataset containing around 30,000 sentences.
* Advanced linguistic features were explored using topic models to arrive at a better system with MAP and MRR scores of 0.68 and 0.69 respectively.
* Additionaly, the system was also equiped with a question prediction module by using a graph based user pattern analysis scheme.  
* A multi-class question type classifier was built to allow the model to further extract question type specific features.
* A classic IR method involving tf-idf weight vectors was used for retrieving relevant documents for each question asked.
* K-Means and hierarchical clustering algorithms were used to group common documents and thereby aid in reducing the search space.


###Folders
* Data - Contains the initial WikiQA dataset used in the research and also the dataset at each checkpoint in the processing step.

* Models - Contains the various models that were built for the project. Includes the following:
..* IR - Contains the clustering models, cluster centers, tf-idf vocabulary and tf-idf vectorizer used.
..* LDA - Contains the topic vectorizer, count vectorizer to get feature vectors and average positive and negative topic vectors.
..* QA Classifier - Contains the two logistic regression binary classifiers that are to be used for answer classification.
..* Question Classifier - Contains the models built for question type classification.

* Code - Contains the ipython notebooks that were used during the processing step and for building the models.
..* Question Classification - Contains the glove.6B.50d word vectors and the necessary code for question classification.
..* One, Two - Contain the ipython notebooks and rest of the code.
..* Application - Contains the code for setting up the application environment and running the QA system.

* Report - Contains the final report, presentation and the the latex project of the thesis along with the plagiarism report.


###Running the code
* Download and install all the necessary libraries.
* Open terminal
* Change directory to 

>	"/Code/Application"
* Run the app.py file using the command 

>	python app.py
* On your browser, go to 

>	http://localhost:5001/index
* The application frontend is displayed with a textbox to enter the question and a submit button to get the answer


_Note : Update file paths._
