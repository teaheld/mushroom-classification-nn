import numpy as np
import pandas as pd

from functools import partial
norm = partial(np.linalg.norm, ord = 1, axis = -1)


def complement_code(x):
	x = [x, 1 - x]
	return pd.concat(x, axis = 1)

def fuzzy_and(new_weights, old_weights):
	res = []
	for i in range(len(new_weights)):
		res.append(min(new_weights[i], old_weights[i]))
	return res

class Neuron:
	def __init__(self, size):
		self.weights = np.ones(size)
		self.state = 0
		self.label = -1
	
	def set_label(self, label):
		self.label = label

	def set_weights(self, weights):
		self.weights = weights

	def calculate_choice_func(self, input, alpha):
		return norm(fuzzy_and(input, self.weights)) / (alpha + norm(self.weights))
		

	def calculate_resonance_condition(self, input, rho):
		return norm(fuzzy_and(input, self.weights)) / norm(input)

	def update_weights_first(self, input, label):
		self.weights = input
		self.label = label
		self.state = 1


	def update_weights(self, input, beta):
		self.weights = list(map(lambda x, y: beta * x + (1 - beta) * y, fuzzy_and(input, self.weights), self.weights))
		#print(self.weights)

	def class_match(self, label):
		return 1 if self.label == label else 0


class NeuralNetwork:
	def __init__(self, input_size, rho, alpha, beta, add_rho, epochs):
		self.input_size = input_size
		self.neurons = []
		self.neurons.append(Neuron(self.input_size))
		self.rho = rho
		self.alpha = alpha
		self.beta = beta
		self.add_rho = add_rho
		self.epochs = epochs

	def train(self, data, labels):
		training_labels = np.ones(labels.size)
		training_labels = - training_labels
			
		#for e in range(self.epochs):
		for i in range(len(data)):
			# Set the vigilane factor equal to its baseline value.
			rho = self.rho
			# Choice functions.
			Tj = []
			# For each available neuron calculate choice function. At the beginning we have one neuron, which is uncommitted.
			for j in range(len(self.neurons)):
				Tj.append((j, self.neurons[j].calculate_choice_func(data[i], self.alpha)))

			# Find the winner neuron.
			Tj = sorted(Tj, key=lambda item:item[1], reverse = True)

			while len(Tj) != 0 :
				(j, tj) = Tj.pop(0)
				# First we check if the winner neuron is uncommited.
				if self.neurons[j].state == 0:
					# If so, we assign the input vector as the weight vector of the winner neuron and
					# we set the class label of winner neuron to be the class label of the input vector.
					self.neurons[j].update_weights_first(data[i], labels[i])
					training_labels[i] = self.neurons[j].label

					# Now we create a new uncommitted neuron to check its properties with next input.
					self.neurons.append(Neuron(self.input_size))
					# We've found the right label, so we take the next input.
					break
					
				# If the winner neuron is committed,
				else:
					# we check the resonance condition, or if the input is similar enough to the winner's prototype.
					Cj = self.neurons[j].calculate_resonance_condition(data[i], rho)
					# If so,
					if Cj >= rho:
						# we have to check if the input and the winner neuron have the same class labels.
						if self.neurons[j].class_match(labels[i]) == 1:
							# If that's true, we update the winner neuron to be closer to the input.
							self.neurons[j].update_weights(data[i], self.beta)
							training_labels[i] = self.neurons[j].label
							# We've found the right label, so we take the next input.
							break

						# If classes do not match, we try the next winner.
						else:
							# But now we increase the vigilance factor.
							rho = Cj + self.add_rho	
							#continue
							# If vigilance factor is larger than 1, we terminate the training for this input 
							# in the current epoch.
							if rho > 1:
								break

					# If the input is not similar enough to the winner, we check the next neuron in the list.

		return training_labels

	def test(self, data, labels):
		test_labels = np.ones(len(labels))
		test_labels = - test_labels

		# For each input	
		for i in range(len(data)):
			Tj = []
			# Calculate choice functions.
			for j in range(len(self.neurons)):
				Tj.append((j, self.neurons[j].calculate_choice_func(data[i], self.alpha)))

			# Find the winner neuron.
			Tj = sorted(Tj, key=lambda item:item[1], reverse = True)
			(j, tj) = Tj.pop(0)
	
			# Set the label of winner neuron to be the class of the input data.
			test_labels[i] = self.neurons[j].label
		
		return test_labels



from sklearn.model_selection import train_test_split

data = pd.read_csv("../tea/data/mushrooms.csv")

# Veil-type only has value 'p', so we remove this attribute, because it has no effect on classification.
data = data.drop("veil-type", axis = 1)

data['class'].replace('p', 0, inplace = True)
data['class'].replace('e', 1, inplace = True)

# We split data and labels.
labels = data.loc[: , ['class']]
data = data.drop(["class"], axis = 1)



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for column in data.columns:
    data[column] = labelencoder.fit_transform(data[column])

# Normalizing data. The data now is in interval [0, 1].
data = (data - data.min()) / (data.max() - data.min())

# Calculate complement code. Now the input has size 2 * d.
data = complement_code(data)

# Split the data into training and test data.
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
data_train = data_train.values
data_test = data_test.values
labels_train = labels_train.values
labels_test = labels_test.values

# Now we create a fuzzy neural network. We set the vigilance factor to 1, alpha to 0.001, beta to 1 and epsilon to 0.001. We try 1 epoch.
net = NeuralNetwork(data_train.shape[1], 1, 0.001, 1, 0.001, 1)

'''
rho = 0
alpha = 0.001
beta = 1
eps = 0.001

data_train = data_train[:100]
labels_train = labels_train[:100]

Training accuracy:  100.0
Testing accuracy:  94.99589827727645
Neurons: 6

rho = 0.25
Training accuracy:  100.0
6
Testing accuracy:  94.99589827727645

rho = 0.5
Training accuracy:  100.0
Testing accuracy:  97.33388022969646
Neurons: 10

rho = 0.75
Training accuracy:  100.0
18
Testing accuracy:  97.00574241181296

rho = 1
Training accuracy:  100.0
100
Testing accuracy:  96.5135356849877


1000 samples
rho = 0
Training accuracy:  100.0
7
Testing accuracy:  99.87694831829369

rho = 0.25
Training accuracy:  100.0
9
Testing accuracy:  99.87694831829369

rho = 0.5
Training accuracy:  100.0
16
Testing accuracy:  99.54881050041017

Training accuracy:  100.0
26
Testing accuracy:  99.79491386382281

Training accuracy:  100.0
1000
Testing accuracy:  99.79491386382281


whole sample
rho = 0
Training accuracy:  100.0
8
Testing accuracy:  100.0

rho = 0.25
Training accuracy:  100.0
10
Testing accuracy:  100.0

rho = 0.5
Training accuracy:  100.0
16
Testing accuracy:  100.0

rho = 0.75
Training accuracy:  100.0
26
Testing accuracy:  100.0

train = 2500
rho = 0
Training accuracy:  100.0
7
Testing accuracy:  99.87694831829369

rho = 0.25
Training accuracy:  100.0
9
Testing accuracy:  99.87694831829369

rho = 0.5
Training accuracy:  100.0
16
Testing accuracy:  99.87694831829369

rho = 0.75
Training accuracy:  100.0
26
Testing accuracy:  99.91796554552911

train size = 3500
Training accuracy:  100.0
8
Testing accuracy:  100.0

'''
'''
data_train = data_train
labels_train = labels_train'''
# We train the network.
train_labels = net.train(data_train, labels_train)
count_train_hits = 0
for i in range(len(labels_train)):
	if train_labels[i] == labels_train[i]:
		count_train_hits += 1

# We remove the last uncommitted neuron so it can't be chosen in the test
if net.neurons[- 1].state == 0:
	del net.neurons[-1]

print("Training accuracy: ", count_train_hits / len(labels_train) * 100)

print(len(net.neurons))

# We test some data:
test_labels = net.test(data_test, labels_test)
count_test_hits = 0
for i in range(len(labels_test)):
	if test_labels[i] == labels_test[i]:
		count_test_hits += 1

print("Testing accuracy: ", count_test_hits / len(labels_test) * 100)

