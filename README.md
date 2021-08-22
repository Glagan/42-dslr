# dslr

## Requirements

``Python 3``, ``pandas``, ``numpy`` and ``matplotlib``.

You can install all dependencies with ``python3 -m pip install -r requirements.txt``.

## Usage

The following scripts can take a single argument, the dataset that will be used, and it always defaults to the one expected by the subject (``dataset_train``).  
Each plots have their values separated to match the house they belongs to.

* ``describe.py``
	* Display statistics about each numeric features.
* ``histogram.py``
	* Display an histogram for each numeric features.
* ``scatter_plot.py``
	* Display the dispersion of each values for each numeric features.
* ``pair_plot.py``
	* Display a scatter matrix of all numeric features.

The main script which generate the *weights* and use a normal Gradient Descent ``logreg_train.py``.  
A plot with the cost over the number of iterations is displayed to show that the algorithm is working.

With the generated weights (in ``thetas.csv``) you can use ``logreg_predict.py`` to predict the houses for the test dataset and save the result in ``houses.csv``.

A bonus script is also included, ``logreg_bonus.py``, which uses a mini-batch stochastic gradient descent algorithm instead of the normal one in the train script.

## Resources

* Course
	* Full Course
		* This is *mandatory*
		* Look at the 42-AI Bootcamp Machine Learning module 8 to see what parts you should watch
		* https://www.coursera.org/learn/machine-learning
* 42-AI
	* https://github.com/42-AI/bootcamp_machine-learning
* Analysis
	* https://en.wikipedia.org/wiki/Percentile#Second_variant,_C_=_1
	* https://en.wikipedia.org/wiki/Standard_deviation
	* https://en.wikipedia.org/wiki/Variance
* Stochastic Gradient Descent
	* https://en.wikipedia.org/wiki/Stochastic_gradient_descent
	* https://optimization.cbe.cornell.edu/index.php?title=Stochastic_gradient_descent
