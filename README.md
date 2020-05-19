


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Seinfeld_logo.svg/1200px-Seinfeld_logo.svg.png" alt="drawing" width="200" style="float: right"/>

# The-App-About-Nothing (www.theappaboutnothing.com)

A web app for Seinfeld enthusiasts! The web app has two functions: The user can change episode features and see how the episode would be rated, the user can choose the starting words and length of phrase and generate lines for AI Jerry!

### Step 1: Collecting the Data

* I was fortunate to find a file that contained most of the seinfeld scripts however I had to webscrape some missing episodes from [this website](http://www.seinfeldscripts.com/) and made a DataFrame in pandas with the rows being every episode and the column being the script
* I webscraped the episode ratings from IMDb and appended them to the DataFrame as the target feature for the model
* I webscraped every location visited from each episode from [this website](https://mapsaboutnothing.com/) to improve my models accuracy

### Step 2: Feature Engineering:

* Using basic python, I was able to count the amount of lines each character (Jerry, George, Elaine, and Kramer) had per episode and appended that as a feature to the DataFrame
* Using NLTK I tokenized each sentence then used the [VADER sentiment analysis](https://github.com/cjhutto/vaderSentiment) to give each sentence a sentiment rating and took the mean of each sentence for each character and appended it as a feature to the DataFrame (mean sentiment varies more than you would think between episodes)
* A list of episode locations (if they were in 5 or more episodes) was appended as a feature to the DataFrame

### Step 3: Training the model

I used a variety of models (random forest, SVM, and gradient boosting) but the gradient boost model worked the best. I used a grid search over multiple paramters to optimize for the RMSE.

### Step 4: Make a LSTM Neural Network

Deep learning is so powerful and definitley an area of interest to me so I decided to try and train a LSTM (powerful flavor of RNN) to talk like Jerry Seinfeld would. I used a character based approach from every line Jerry says across all episodes (I would recommend using a bigger dataset for better results). After hours of research and fine tuning and even more hours of training on [Google Colab (GPU)](https://colab.research.google.com) I was able to get decently sensible english out of AI Jerry


### Step 5: Host Everything on a Server so Anyone Can Use it

* I used Flask, a Python “microframework” to develop the web application.
* I used Docker to port my application stack all with all its dependencies (TensorFlow model and pickled Gradient Boost model) and getting it to run on a server with no problems
* Everything is hosted on a server provided by [Digital Ocean](https://www.digitalocean.com/) as they have great prices for small server packages


I hope you enjoy creating episodes to be rated using the various features and I also hope you get a kick out of what AI Jerry says to you!
