# Variational Quantum Classifier (VQC) Implementation in Python using PennyLane for the Parity Problem

# Contributors
    Vasileios Katsaitis, AM: 1115202000073

# Description
In this assignment, our initial task is to create a **variational quantum circuit**, the output of which will be used for our quantum classifier model. As previously stated, we will be using the PennyLane library, in order to simplify the process of creating powerful quantum circuits. We will, initially, run this model with the functions and hyperparameter values mentioned below, however we will, later on, experiment with different functions and metrics, as well as different inputs (data in R5). We will use our classifier to try to solve the **parity problem**, initially for R3 and later on for R5 data. We will, therefore, experiment with different model hyperparameters, such as number of qubits, layers, epochs, step size, etc, as well as optimizers and loss functions, to determine what the best performing model for our classification needs is. Furthermore, we will carry out additional tasks, such as including parameter shifting in our training process, plotting Bloch spheres, and comparing our model with a simple classical Neural Network one.

# Conventions
- **Circuit**: We use **Amplitude Encoding (Embedding)** to prepare our quantum states for our circuit. This effectively receives floating point data (representing points of R3 or R5) and creates an amplitude vector of 'n_qubits' qubits which is used as input in respective gates. We use an arbitrary single qubit **rotation gate** (calculated using the RY and RZ gates), **CNOT** gates for entanglement and a **Pauli-Z** gate for our measurements (since this is a binary classification task, we only get the expected value of the first qubit). It returns only the measurement for the first qubit, if we use parameter shifting, or both the measurement and the quantum state, if we do not. The returned quantum state will be extremely useful, when it comes to Bloch Sphere plotting. Note that we only use the first qubit for our measurements, since we do not need other measurements for this binary classification task.  
- **Classifier**: Simply returns the output of the prementioned circuit, plus a bias term.
- **(Default) Cost function**: We use the **binary cross entropy** function, which is widely used in binary classification tasks. We choose this loss function intuitively, since our quantum classifier constitutes a probabilistic model, in a sense.
- **(Default) Optimizer**: We use the **Adam Optimizer** with the default step size of 0.01, which is a more efficient Gradient Descent algorithm with an adaptive learning rate, proven to converge faster than other variations of the (Gradient Descent) algorithm. 
- **Classification Results**: In the model training function 'train_VQC' we print out, on every iteration, the epoch (iteration) we're in and the respective training and testing data accuracy and loss scores.

# Hyperparameters and their Default Values
- **Number of qubits (n_qubits)** = 3
- **Number of layers (num_layers)** = 2
- **Epochs** = 20
- **Batch size** = 10
- **Step size** = 0.01
- **Bias term** = 0

We can always experiment with different hyperparameter values, which we will do later on. It's also **important** to mention the following: 
- We don't strictly specify the **number of epochs** used, but we're interchangeably using either 20 or 30 epochs (depending on how quickly model converges towards an optimal solution). Models that approach optimal solution early on will NOT exceed a number of 20 epochs, whereas models for which convergence occurs at a slower rate will be able to reach 30 epochs. This is a convention made for every model comparison. We can use a higher number of epochs, if need be, but it will mostly lead to higher execution time, which might not be necessary.
- **Batch size** does **NOT** refer to data batches we used for accuracy and loss score calculations, but rather batches used in order to calculate **optimal model parameters and bias term** inside of the 'train_VQC' function. We calculate every accuracy and loss score using our entire dataset.
- In some instances, we mention data **clipping**. Clipping is a method we apply in some instances on the predicted labels array, which we pass onto the respective cost function, in order to avoid instances of log(0) or log(1) from occuring. It is practically useful whenever we use the 'binary_cross_entropy' function with some optimizers, like the Nesterov Momentum Optimizer, which we will mention later on.


# Task 2 Topics
This section corresponds to requested research and model comparisons needed for the second task of the assignment. Our goal, here, is to appropriately respond to every subtopic of the list of 5 different provided topics.

## 1. Classical optimizations and loss functions
In this subsection, we compare our model's efficiency by modifying it appropriately. This includes tuning some hyperparameters, as well as trying out different optimizer and/or loss functions. For that purpose, we run our model on the default scenario first and, then, experiment with some modifications, to see if we achieve better accuracy and/or loss scores. We, essentially, compare our model training results with one another, just to point out possible improvements. According to the result we get, we finally choose the model that looks more "promising" than others and tweak some hyperparameters to elevate its performance.

## Method A: Hyperparameter tuning using default optimizer

### Default
This is our (average case) model output using **default hyperparameters** and the **Adam Optimizer**:

- Epoch 1 --- Training Accuracy: 23.58% --- Testing Accuracy: 29.17% --- Training Loss: 1.0129 --- Testing Loss: 0.9856
- Epoch 2 --- Training Accuracy: 23.43% --- Testing Accuracy: 27.98% --- Training Loss: 0.9917 --- Testing Loss: 0.9670
- Epoch 3 --- Training Accuracy: 23.80% --- Testing Accuracy: 27.58% --- Training Loss: 0.9716 --- Testing Loss: 0.9493
- Epoch 4 --- Training Accuracy: 25.57% --- Testing Accuracy: 27.58% --- Training Loss: 0.9521 --- Testing Loss: 0.9321
- Epoch 5 --- Training Accuracy: 25.65% --- Testing Accuracy: 27.58% --- Training Loss: 0.9336 --- Testing Loss: 0.9158
- Epoch 6 --- Training Accuracy: 25.87% --- Testing Accuracy: 29.76% --- Training Loss: 0.9157 --- Testing Loss: 0.9001
- Epoch 7 --- Training Accuracy: 45.31% --- Testing Accuracy: 46.83% --- Training Loss: 0.8983 --- Testing Loss: 0.8850
- Epoch 8 --- Training Accuracy: 47.23% --- Testing Accuracy: 46.23% --- Training Loss: 0.8822 --- Testing Loss: 0.8710
- Epoch 9 --- Training Accuracy: 47.30% --- Testing Accuracy: 46.43% --- Training Loss: 0.8668 --- Testing Loss: 0.8577
- Epoch 10 --- Training Accuracy: 47.60% --- Testing Accuracy: 46.03% --- Training Loss: 0.8529 --- Testing Loss: 0.8457
- Epoch 11 --- Training Accuracy: 49.22% --- Testing Accuracy: 46.23% --- Training Loss: 0.8395 --- Testing Loss: 0.8341
- Epoch 12 --- Training Accuracy: 48.34% --- Testing Accuracy: 46.03% --- Training Loss: 0.8270 --- Testing Loss: 0.8234
- Epoch 13 --- Training Accuracy: 48.12% --- Testing Accuracy: 45.44% --- Training Loss: 0.8162 --- Testing Loss: 0.8139
- Epoch 14 --- Training Accuracy: 48.04% --- Testing Accuracy: 45.44% --- Training Loss: 0.8056 --- Testing Loss: 0.8048
- Epoch 15 --- Training Accuracy: 48.78% --- Testing Accuracy: 46.83% --- Training Loss: 0.7952 --- Testing Loss: 0.7958
- Epoch 16 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.81% --- Training Loss: 0.7854 --- Testing Loss: 0.7874
- Epoch 17 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.61% --- Training Loss: 0.7759 --- Testing Loss: 0.7793
- Epoch 18 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.21% --- Training Loss: 0.7667 --- Testing Loss: 0.7715
- Epoch 19 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.21% --- Training Loss: 0.7580 --- Testing Loss: 0.7642
- Epoch 20 --- Training Accuracy: 51.15% --- Testing Accuracy: 48.21% --- Training Loss: 0.7492 --- Testing Loss: 0.7568

We can see that both training and testing accuracy predominantly increases, as the number of epochs increases, whereas respective loss scores decrease. This means that our model is slowly, but steadily, converging towards an optimal solution.

### Batch size = 5
This model uses the exact same hyperparameters as those of the default scenario, apart from 'batch_size', which is set to 5:

- Epoch 1 --- Training Accuracy: 23.58% --- Testing Accuracy: 29.17% --- Training Loss: 1.0129 --- Testing Loss: 0.9856
- Epoch 2 --- Training Accuracy: 23.43% --- Testing Accuracy: 27.98% --- Training Loss: 0.9917 --- Testing Loss: 0.9670
- Epoch 3 --- Training Accuracy: 23.80% --- Testing Accuracy: 27.58% --- Training Loss: 0.9716 --- Testing Loss: 0.9493
- Epoch 4 --- Training Accuracy: 25.57% --- Testing Accuracy: 27.58% --- Training Loss: 0.9521 --- Testing Loss: 0.9321
- Epoch 5 --- Training Accuracy: 25.65% --- Testing Accuracy: 27.58% --- Training Loss: 0.9336 --- Testing Loss: 0.9158
- Epoch 6 --- Training Accuracy: 25.87% --- Testing Accuracy: 29.76% --- Training Loss: 0.9157 --- Testing Loss: 0.9001
- Epoch 7 --- Training Accuracy: 45.31% --- Testing Accuracy: 46.83% --- Training Loss: 0.8983 --- Testing Loss: 0.8850
- Epoch 8 --- Training Accuracy: 47.23% --- Testing Accuracy: 46.23% --- Training Loss: 0.8822 --- Testing Loss: 0.8710
- Epoch 9 --- Training Accuracy: 47.30% --- Testing Accuracy: 46.43% --- Training Loss: 0.8668 --- Testing Loss: 0.8577
- Epoch 10 --- Training Accuracy: 47.60% --- Testing Accuracy: 46.03% --- Training Loss: 0.8529 --- Testing Loss: 0.8457
- Epoch 11 --- Training Accuracy: 49.22% --- Testing Accuracy: 46.23% --- Training Loss: 0.8395 --- Testing Loss: 0.8341
- Epoch 12 --- Training Accuracy: 48.34% --- Testing Accuracy: 46.03% --- Training Loss: 0.8270 --- Testing Loss: 0.8234
- Epoch 13 --- Training Accuracy: 48.12% --- Testing Accuracy: 45.44% --- Training Loss: 0.8162 --- Testing Loss: 0.8139
- Epoch 14 --- Training Accuracy: 48.04% --- Testing Accuracy: 45.44% --- Training Loss: 0.8056 --- Testing Loss: 0.8048
- Epoch 15 --- Training Accuracy: 48.78% --- Testing Accuracy: 46.83% --- Training Loss: 0.7952 --- Testing Loss: 0.7958
- Epoch 16 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.81% --- Training Loss: 0.7854 --- Testing Loss: 0.7874
- Epoch 17 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.61% --- Training Loss: 0.7759 --- Testing Loss: 0.7793
- Epoch 18 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.21% --- Training Loss: 0.7667 --- Testing Loss: 0.7715
- Epoch 19 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.21% --- Training Loss: 0.7580 --- Testing Loss: 0.7642
- Epoch 20 --- Training Accuracy: 51.15% --- Testing Accuracy: 48.21% --- Training Loss: 0.7492 --- Testing Loss: 0.7568

We can see that our model's performance didn't change at all, when changing the batch size to 5 instead of 10. Essentially, this hyperparameter is solely used to define optimal parameters and bias term to be used in accuracy and cost function calculations. A smaller batch size number might not lead to any significant differencies, on that case. 

### Step size = 0.05
This model uses the exact same hyperparameters as those of the default scenario, apart from 'step_size', which is set to 0.05:

- Epoch 1 --- Training Accuracy: 25.65% --- Testing Accuracy: 27.58% --- Training Loss: 0.9897 --- Testing Loss: 0.9559
- Epoch 2 --- Training Accuracy: 48.34% --- Testing Accuracy: 46.03% --- Training Loss: 0.9084 --- Testing Loss: 0.8831
- Epoch 3 --- Training Accuracy: 51.07% --- Testing Accuracy: 48.81% --- Training Loss: 0.8417 --- Testing Loss: 0.8237
- Epoch 4 --- Training Accuracy: 51.15% --- Testing Accuracy: 48.41% --- Training Loss: 0.7864 --- Testing Loss: 0.7750
- Epoch 5 --- Training Accuracy: 51.00% --- Testing Accuracy: 48.41% --- Training Loss: 0.7426 --- Testing Loss: 0.7369
- Epoch 6 --- Training Accuracy: 44.12% --- Testing Accuracy: 43.85% --- Training Loss: 0.7086 --- Testing Loss: 0.7071
- Epoch 7 --- Training Accuracy: 25.20% --- Testing Accuracy: 24.60% --- Training Loss: 0.6943 --- Testing Loss: 0.6942
- Epoch 8 --- Training Accuracy: 7.46% --- Testing Accuracy: 9.52% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 9 --- Training Accuracy: 7.98% --- Testing Accuracy: 9.72% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 10 --- Training Accuracy: 8.28% --- Testing Accuracy: 9.92% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 11 --- Training Accuracy: 8.28% --- Testing Accuracy: 9.92% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 12 --- Training Accuracy: 8.28% --- Testing Accuracy: 9.92% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 13 --- Training Accuracy: 8.28% --- Testing Accuracy: 9.92% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 14 --- Training Accuracy: 8.28% --- Testing Accuracy: 9.92% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 15 --- Training Accuracy: 8.28% --- Testing Accuracy: 10.12% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 16 --- Training Accuracy: 9.16% --- Testing Accuracy: 10.71% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 17 --- Training Accuracy: 9.31% --- Testing Accuracy: 11.11% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 18 --- Training Accuracy: 16.26% --- Testing Accuracy: 18.85% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 19 --- Training Accuracy: 20.99% --- Testing Accuracy: 24.60% --- Training Loss: 0.6931 --- Testing Loss: 0.6931
- Epoch 20 --- Training Accuracy: 20.99% --- Testing Accuracy: 24.60% --- Training Loss: 0.6931 --- Testing Loss: 0.6931

This optimization technique involves increasing the step size, so at to converge slightly quicker to an optimal solution (reducing step size leads to slower, but more stable, convergence). We can see that both testing and training accuracy scores have increased, starting to as early as the second epoch. Loss scores also descrease, with every iteration. However, when reaching the 5th epoch, we see that model accuracy is starting to decrease, plummeting at the 8th epoch. At the same time, after the 8th epoch, we see that loss function has reached a global minimum and is not changing at all. This is because the algorithm has already converged towards an optimal solution. It's **important** to note, at this point, that this optimization occured, because we clip predicted labels inside of the 'binary_cross_entropy' function by a small epsilon value (1e-12). We also did **NOT** apply any data clipping on the default scenario. The clipping line of code can be seen inside of the 'binary_cross_entropy' function (which is obviously commented out whenever we DON'T mention we've clipped our data).

### Number of qubits = 4, number of layers = 3
This model uses the exact same hyperparameters as those of the default scenario, apart from 'n_qubits' and 'num_layers', set to 4 and 3 respectively:

- Epoch 1 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9916 --- Testing Loss: 0.9570
- Epoch 2 --- Training Accuracy: 78.88% --- Testing Accuracy: 75.40% --- Training Loss: 0.9643 --- Testing Loss: 0.9323
- Epoch 3 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9389 --- Testing Loss: 0.9091
- Epoch 4 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9146 --- Testing Loss: 0.8872
- Epoch 5 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8944 --- Testing Loss: 0.8691
- Epoch 6 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8749 --- Testing Loss: 0.8516
- Epoch 7 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8561 --- Testing Loss: 0.8347
- Epoch 8 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8383 --- Testing Loss: 0.8189
- Epoch 9 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8230 --- Testing Loss: 0.8052
- Epoch 10 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8081 --- Testing Loss: 0.7921
- Epoch 11 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7941 --- Testing Loss: 0.7797
- Epoch 12 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7798 --- Testing Loss: 0.7670
- Epoch 13 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7655 --- Testing Loss: 0.7544
- Epoch 14 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7514 --- Testing Loss: 0.7420
- Epoch 15 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7373 --- Testing Loss: 0.7297
- Epoch 16 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7240 --- Testing Loss: 0.7180
- Epoch 17 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.7106 --- Testing Loss: 0.7064
- Epoch 18 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.6986 --- Testing Loss: 0.6960
- Epoch 19 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.6866 --- Testing Loss: 0.6856
- Epoch 20 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.6750 --- Testing Loss: 0.6757

This optimization results in way better accuracy scores, as well as bigger decreases in loss scores. Algorithm converges quickly to a maximum accuracy score with a constantly descreasing loss score.

### Number of qubits = 4, number of layers = 3, step size = 0.001
One idea to counteract problem above is to decrease initial value of 'step_size' to 0.001 AND increase number of epochs to a number bigger than 20, for instance 30. This should lead to a slower, but more steady convergence towards an optimal solution (even though loss function will not reach its global minimum):

- Epoch 1 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 1.0180 --- Testing Loss: 0.9811
- Epoch 2 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 1.0151 --- Testing Loss: 0.9785
- Epoch 3 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0122 --- Testing Loss: 0.9758
- Epoch 4 --- Training Accuracy: 71.59% --- Testing Accuracy: 70.26% --- Training Loss: 1.0094 --- Testing Loss: 0.9733
- Epoch 5 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0069 --- Testing Loss: 0.9710
- Epoch 6 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0044 --- Testing Loss: 0.9688
- Epoch 7 --- Training Accuracy: 71.72% --- Testing Accuracy: 70.36% --- Training Loss: 1.0019 --- Testing Loss: 0.9665
- Epoch 8 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.9994 --- Testing Loss: 0.9642
- Epoch 9 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9971 --- Testing Loss: 0.9621
- Epoch 10 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9948 --- Testing Loss: 0.9600
- Epoch 11 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9925 --- Testing Loss: 0.9579
- Epoch 12 --- Training Accuracy: 71.98% --- Testing Accuracy: 70.36% --- Training Loss: 0.9901 --- Testing Loss: 0.9558
- Epoch 13 --- Training Accuracy: 73.26% --- Testing Accuracy: 70.96% --- Training Loss: 0.9876 --- Testing Loss: 0.9535
- Epoch 14 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.9851 --- Testing Loss: 0.9512
- Epoch 15 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.9825 --- Testing Loss: 0.9489
- Epoch 16 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.16% --- Training Loss: 0.9800 --- Testing Loss: 0.9465
- Epoch 17 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9773 --- Testing Loss: 0.9441
- Epoch 18 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9748 --- Testing Loss: 0.9418
- Epoch 19 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9722 --- Testing Loss: 0.9394
- Epoch 20 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9695 --- Testing Loss: 0.9370
- Epoch 21 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9669 --- Testing Loss: 0.9346
- Epoch 22 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9643 --- Testing Loss: 0.9322
- Epoch 23 --- Training Accuracy: 73.56% --- Testing Accuracy: 71.16% --- Training Loss: 0.9615 --- Testing Loss: 0.9297
- Epoch 24 --- Training Accuracy: 78.97% --- Testing Accuracy: 75.30% --- Training Loss: 0.9589 --- Testing Loss: 0.9273
- Epoch 25 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9562 --- Testing Loss: 0.9249
- Epoch 26 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9536 --- Testing Loss: 0.9224
- Epoch 27 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9509 --- Testing Loss: 0.9200
- Epoch 28 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9483 --- Testing Loss: 0.9176
- Epoch 29 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9458 --- Testing Loss: 0.9154
- Epoch 30 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9433 --- Testing Loss: 0.9131

We notice that there are no spikes in current model execution, because of the slower convergence. This model achieves the best accuracy scores possible, so we consider it our **best model** so far.

## Method B: Different optimizer functions using best hyperparameters
Although the Adam optimizer is proven to lead to faster convergence, we can experiment with different classical optimizers. For that purpose, whenever we're comparing our quantum classifier results on these different models, we're going to be using the hyperparameters of the **best** model so far (what we call, best hyperparameters). Best model, on that case, is the one with 4 qubits, 3 layers and a step size of 0.001. With that being said, we now present more classical optimization methods, by simply changing the optimization function.  

### Nesterov Momentum Optimizer
This optimization algorithm uses momentum to accelarate gradient vectors in the right directions, leading to the faster convergence we can see below (for epochs = 20):

- Epoch 1 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 1.0183 --- Testing Loss: 0.9815
- Epoch 2 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0138 --- Testing Loss: 0.9773
- Epoch 3 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0082 --- Testing Loss: 0.9722
- Epoch 4 --- Training Accuracy: 71.72% --- Testing Accuracy: 70.36% --- Training Loss: 1.0004 --- Testing Loss: 0.9651
- Epoch 5 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.9931 --- Testing Loss: 0.9585
- Epoch 6 --- Training Accuracy: 72.92% --- Testing Accuracy: 70.96% --- Training Loss: 0.9852 --- Testing Loss: 0.9512
- Epoch 7 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.06% --- Training Loss: 0.9766 --- Testing Loss: 0.9434
- Epoch 8 --- Training Accuracy: 73.52% --- Testing Accuracy: 71.16% --- Training Loss: 0.9680 --- Testing Loss: 0.9355
- Epoch 9 --- Training Accuracy: 78.97% --- Testing Accuracy: 75.40% --- Training Loss: 0.9600 --- Testing Loss: 0.9283
- Epoch 10 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9517 --- Testing Loss: 0.9208
- Epoch 11 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9435 --- Testing Loss: 0.9133
- Epoch 12 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9347 --- Testing Loss: 0.9053
- Epoch 13 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9253 --- Testing Loss: 0.8968
- Epoch 14 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9155 --- Testing Loss: 0.8879
- Epoch 15 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9053 --- Testing Loss: 0.8787
- Epoch 16 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8952 --- Testing Loss: 0.8696
- Epoch 17 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8847 --- Testing Loss: 0.8600
- Epoch 18 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8749 --- Testing Loss: 0.8512
- Epoch 19 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8647 --- Testing Loss: 0.8420
- Epoch 20 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.8545 --- Testing Loss: 0.8329

With this optimization, algorithm approaches the optimal solution faster, right from the 10th epoch. So, the selected number of epochs of this model is adequate to approach the optimal solution. It's, again, **important** to note, here, that we've clipped predicted labels here are well, to avoid runtime errors and inaccurate results. We're also not using 30 epochs here, as opposed to our best model, since accuracy scores do not get better after the 20th epoch.

### Adagrad Optimizer
The Adagrad Optimizer adapts the learning rate (step size) for each parameter based on how frequently they get updated during training. Simply put, the more updates a parameter receives, the smaller the updates. Let's test out our best model with this optimizer, for 30 epochs:

- Epoch 1 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 1.0180 --- Testing Loss: 0.9811
- Epoch 2 --- Training Accuracy: 71.29% --- Testing Accuracy: 70.06% --- Training Loss: 1.0160 --- Testing Loss: 0.9794
- Epoch 3 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 1.0148 --- Testing Loss: 0.9782
- Epoch 4 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0130 --- Testing Loss: 0.9766
- Epoch 5 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0128 --- Testing Loss: 0.9764
- Epoch 6 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0119 --- Testing Loss: 0.9756
- Epoch 7 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 1.0110 --- Testing Loss: 0.9748
- Epoch 8 --- Training Accuracy: 71.51% --- Testing Accuracy: 70.26% --- Training Loss: 1.0103 --- Testing Loss: 0.9741
- Epoch 9 --- Training Accuracy: 71.59% --- Testing Accuracy: 70.26% --- Training Loss: 1.0101 --- Testing Loss: 0.9739
- Epoch 10 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0093 --- Testing Loss: 0.9733
- Epoch 11 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0087 --- Testing Loss: 0.9727
- Epoch 12 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0078 --- Testing Loss: 0.9718
- Epoch 13 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0069 --- Testing Loss: 0.9710
- Epoch 14 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0059 --- Testing Loss: 0.9701
- Epoch 15 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0050 --- Testing Loss: 0.9693
- Epoch 16 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0044 --- Testing Loss: 0.9687
- Epoch 17 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0035 --- Testing Loss: 0.9679
- Epoch 18 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0031 --- Testing Loss: 0.9675
- Epoch 19 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0022 --- Testing Loss: 0.9667
- Epoch 20 --- Training Accuracy: 71.68% --- Testing Accuracy: 70.26% --- Training Loss: 1.0015 --- Testing Loss: 0.9661
- Epoch 21 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 1.0008 --- Testing Loss: 0.9655
- Epoch 22 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 1.0001 --- Testing Loss: 0.9648
- Epoch 23 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.9992 --- Testing Loss: 0.9640
- Epoch 24 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.9988 --- Testing Loss: 0.9636
- Epoch 25 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9982 --- Testing Loss: 0.9630
- Epoch 26 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9975 --- Testing Loss: 0.9624
- Epoch 27 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9969 --- Testing Loss: 0.9619
- Epoch 28 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9965 --- Testing Loss: 0.9615
- Epoch 29 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9962 --- Testing Loss: 0.9612
- Epoch 30 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9957 --- Testing Loss: 0.9607

We can see that algorithm converges towards an optimal on a slower mannter, with best accuracy score not reaching the maximum one that's been found. Convergence becomes slow, as accuracy scores barely change, whereas the loss function exhibits its normal behavior.

### RMSPropOptimizer
This optimizer uses an adaptive learning rate method, like the Adam optimizer, that divides the learning rate for a weight by a moving average of the magnitudes of recent gradients for that specific weight. Essentially, it constitutes an improvement of the Adagrad optimizer, as we will see below, that tackles the issue of learning rate decay by emphasizing on recent gradient values. This is our new model performance for 30 epochs:

- Epoch 1 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 1.0115 --- Testing Loss: 0.9752
- Epoch 2 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0054 --- Testing Loss: 0.9697
- Epoch 3 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 1.0012 --- Testing Loss: 0.9657
- Epoch 4 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9956 --- Testing Loss: 0.9607
- Epoch 5 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.9948 --- Testing Loss: 0.9601
- Epoch 6 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.9916 --- Testing Loss: 0.9571
- Epoch 7 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.9884 --- Testing Loss: 0.9542
- Epoch 8 --- Training Accuracy: 73.26% --- Testing Accuracy: 70.96% --- Training Loss: 0.9858 --- Testing Loss: 0.9519
- Epoch 9 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.9851 --- Testing Loss: 0.9512
- Epoch 10 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.06% --- Training Loss: 0.9821 --- Testing Loss: 0.9485
- Epoch 11 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9797 --- Testing Loss: 0.9463
- Epoch 12 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9759 --- Testing Loss: 0.9428
- Epoch 13 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9722 --- Testing Loss: 0.9395
- Epoch 14 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9685 --- Testing Loss: 0.9361
- Epoch 15 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.9650 --- Testing Loss: 0.9328
- Epoch 16 --- Training Accuracy: 73.61% --- Testing Accuracy: 71.16% --- Training Loss: 0.9623 --- Testing Loss: 0.9305
- Epoch 17 --- Training Accuracy: 78.97% --- Testing Accuracy: 75.30% --- Training Loss: 0.9586 --- Testing Loss: 0.9270
- Epoch 18 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9570 --- Testing Loss: 0.9255
- Epoch 19 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9535 --- Testing Loss: 0.9223
- Epoch 20 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9506 --- Testing Loss: 0.9197
- Epoch 21 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9476 --- Testing Loss: 0.9170
- Epoch 22 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9447 --- Testing Loss: 0.9143
- Epoch 23 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9408 --- Testing Loss: 0.9108
- Epoch 24 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9389 --- Testing Loss: 0.9091
- Epoch 25 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9363 --- Testing Loss: 0.9067
- Epoch 26 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9332 --- Testing Loss: 0.9038
- Epoch 27 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9308 --- Testing Loss: 0.9016
- Epoch 28 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9290 --- Testing Loss: 0.8999
- Epoch 29 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9275 --- Testing Loss: 0.8986
- Epoch 30 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.9251 --- Testing Loss: 0.8963

We can see that, for the same number of epochs as that of the Adagrad model, algorithm here converges faster than before, reaching global maximum accuracy scores from the 18th epoch. In conclusion, the Nesterov Momentum Optimizer provides us with the fastest convergence (out of all the previously tested optimizers) and, thus, is the optimizer we will use from now on.

## Method C: Different loss functions using optimal model
This method aims to compare models based on their different loss functions. Every model we're comparing will use hyperparameters and optimization function that leads to a faster convergence for the smallest number of epochs:

- **Number of qubits (n_qubits)** = 4
- **Number of layers (num_layers)** = 3
- **Step size** = 0.001
- **Batch size** = 10
- **Bias term** = 0
- **Optimizer** = Nesterov Momentum Optimizer
- **Loss function** = Binary Cross Entropy

Model above is the selected model for our comparisons.

### Mean Squared Error (MSE)
Another commonly used loss function is Mean Squared Error, which ultimately calculates the mean square distance between points. It's commonly used in regression, but can be adapted to binary classification, tasks, if we treat our labels as points. Upon executing the model with said loss function, these are the results we get for 30 epochs:  

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.2084 --- Testing Loss: 0.2148
- Epoch 2 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.2079 --- Testing Loss: 0.2145
- Epoch 3 --- Training Accuracy: 70.99% --- Testing Accuracy: 69.65% --- Training Loss: 0.2074 --- Testing Loss: 0.2141
- Epoch 4 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2066 --- Testing Loss: 0.2135
- Epoch 5 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2061 --- Testing Loss: 0.2131
- Epoch 6 --- Training Accuracy: 71.51% --- Testing Accuracy: 70.26% --- Training Loss: 0.2056 --- Testing Loss: 0.2127
- Epoch 7 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2050 --- Testing Loss: 0.2123
- Epoch 8 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2045 --- Testing Loss: 0.2120
- Epoch 9 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2043 --- Testing Loss: 0.2118
- Epoch 10 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2040 --- Testing Loss: 0.2116
- Epoch 11 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2039 --- Testing Loss: 0.2114
- Epoch 12 --- Training Accuracy: 71.68% --- Testing Accuracy: 70.26% --- Training Loss: 0.2035 --- Testing Loss: 0.2112
- Epoch 13 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.2031 --- Testing Loss: 0.2108
- Epoch 14 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2026 --- Testing Loss: 0.2104
- Epoch 15 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2019 --- Testing Loss: 0.2099
- Epoch 16 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.2012 --- Testing Loss: 0.2094
- Epoch 17 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.2003 --- Testing Loss: 0.2088
- Epoch 18 --- Training Accuracy: 73.26% --- Testing Accuracy: 70.96% --- Training Loss: 0.1997 --- Testing Loss: 0.2084
- Epoch 19 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1989 --- Testing Loss: 0.2078
- Epoch 20 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1981 --- Testing Loss: 0.2072
- Epoch 21 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1972 --- Testing Loss: 0.2066
- Epoch 22 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1963 --- Testing Loss: 0.2060
- Epoch 23 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1952 --- Testing Loss: 0.2052
- Epoch 24 --- Training Accuracy: 73.48% --- Testing Accuracy: 71.16% --- Training Loss: 0.1943 --- Testing Loss: 0.2046
- Epoch 25 --- Training Accuracy: 78.92% --- Testing Accuracy: 75.30% --- Training Loss: 0.1933 --- Testing Loss: 0.2039
- Epoch 26 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1923 --- Testing Loss: 0.2033
- Epoch 27 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1914 --- Testing Loss: 0.2027
- Epoch 28 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1906 --- Testing Loss: 0.2022
- Epoch 29 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1899 --- Testing Loss: 0.2017
- Epoch 30 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1893 --- Testing Loss: 0.2013

Already from the first epoch we notice a massive decrease in loss function scores, in both training and testing data. However, algorithm now seems to reach its maximum training accuracy score at the 19th epoch (and testing at the 18th epoch). Convergence is, therefore, more stable and slow than before, but with way better loss scores. 

### Huber Loss
This loss function combines the best properties of the Mean Squarred Error (MSE) and Mean Absolute Error (MAE) loss functions, since it is less sensitive to outliers (generally) than MSE, while being differentiable. On paper, it could be considered a viable option for this classification task, if the predicted output is treated as a probability or a score that's close to 0 or 1. Here are model's results for 30 epochs: 

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.1043 --- Testing Loss: 0.1075
- Epoch 2 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.1042 --- Testing Loss: 0.1074
- Epoch 3 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.1040 --- Testing Loss: 0.1073
- Epoch 4 --- Training Accuracy: 70.18% --- Testing Accuracy: 68.95% --- Training Loss: 0.1038 --- Testing Loss: 0.1071
- Epoch 5 --- Training Accuracy: 71.08% --- Testing Accuracy: 69.75% --- Training Loss: 0.1037 --- Testing Loss: 0.1070
- Epoch 6 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 0.1036 --- Testing Loss: 0.1069
- Epoch 7 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1034 --- Testing Loss: 0.1068
- Epoch 8 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1033 --- Testing Loss: 0.1067
- Epoch 9 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1032 --- Testing Loss: 0.1067
- Epoch 10 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1031 --- Testing Loss: 0.1066
- Epoch 11 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1031 --- Testing Loss: 0.1066
- Epoch 12 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.1030 --- Testing Loss: 0.1065
- Epoch 13 --- Training Accuracy: 71.51% --- Testing Accuracy: 70.26% --- Training Loss: 0.1029 --- Testing Loss: 0.1064
- Epoch 14 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.1027 --- Testing Loss: 0.1063
- Epoch 15 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.1025 --- Testing Loss: 0.1062
- Epoch 16 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.1023 --- Testing Loss: 0.1060
- Epoch 17 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.1021 --- Testing Loss: 0.1058
- Epoch 18 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.1019 --- Testing Loss: 0.1057
- Epoch 19 --- Training Accuracy: 71.68% --- Testing Accuracy: 70.26% --- Training Loss: 0.1017 --- Testing Loss: 0.1055
- Epoch 20 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.1014 --- Testing Loss: 0.1053
- Epoch 21 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.1012 --- Testing Loss: 0.1051
- Epoch 22 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.1009 --- Testing Loss: 0.1049
- Epoch 23 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.1006 --- Testing Loss: 0.1047
- Epoch 24 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.1003 --- Testing Loss: 0.1045
- Epoch 25 --- Training Accuracy: 72.02% --- Testing Accuracy: 70.56% --- Training Loss: 0.1000 --- Testing Loss: 0.1043
- Epoch 26 --- Training Accuracy: 73.26% --- Testing Accuracy: 71.06% --- Training Loss: 0.0996 --- Testing Loss: 0.1040
- Epoch 27 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.0993 --- Testing Loss: 0.1038
- Epoch 28 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.0990 --- Testing Loss: 0.1036
- Epoch 29 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.06% --- Training Loss: 0.0988 --- Testing Loss: 0.1035
- Epoch 30 --- Training Accuracy: 73.39% --- Testing Accuracy: 71.16% --- Training Loss: 0.0985 --- Testing Loss: 0.1033

Convergence is slower than other models we've previously tested (for instance MSE), since we fail to reach maximum accyracy scores even in the last 3 epochs (for both training and testing data). However, loss scores are much better with this cost function, which is valid considering the prementioned fact that Huber loss is less sensitive to some outlier values, as opposed to MSE. 

### Hinge Loss 
This loss function is predominantly used in Support Vector Machines (SVMs), as an optimization function, in order to increase the margin between data points and the decision boundary. However, in the context of SVMs, it is considered the typical loss function for binary classification tasks. Therefore, it's a solid option to consider when testing out different loss functions in the context of this classification task. With hinge loss as our function, we get these results for 30 epochs:

- Epoch 1 --- Training Accuracy: 68.85% --- Testing Accuracy: 67.74% --- Training Loss: 0.9196 --- Testing Loss: 0.9075
- Epoch 2 --- Training Accuracy: 68.76% --- Testing Accuracy: 67.74% --- Training Loss: 0.9196 --- Testing Loss: 0.9073
- Epoch 3 --- Training Accuracy: 68.76% --- Testing Accuracy: 67.74% --- Training Loss: 0.9194 --- Testing Loss: 0.9071
- Epoch 4 --- Training Accuracy: 68.72% --- Testing Accuracy: 67.64% --- Training Loss: 0.9192 --- Testing Loss: 0.9069
- Epoch 5 --- Training Accuracy: 68.55% --- Testing Accuracy: 67.64% --- Training Loss: 0.9188 --- Testing Loss: 0.9064
- Epoch 6 --- Training Accuracy: 67.52% --- Testing Accuracy: 66.83% --- Training Loss: 0.9184 --- Testing Loss: 0.9059
- Epoch 7 --- Training Accuracy: 66.23% --- Testing Accuracy: 66.12% --- Training Loss: 0.9179 --- Testing Loss: 0.9054
- Epoch 8 --- Training Accuracy: 56.20% --- Testing Accuracy: 57.36% --- Training Loss: 0.9173 --- Testing Loss: 0.9047
- Epoch 9 --- Training Accuracy: 56.02% --- Testing Accuracy: 57.16% --- Training Loss: 0.9166 --- Testing Loss: 0.9039
- Epoch 10 --- Training Accuracy: 55.30% --- Testing Accuracy: 56.25% --- Training Loss: 0.9158 --- Testing Loss: 0.9030
- Epoch 11 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.15% --- Training Loss: 0.9150 --- Testing Loss: 0.9020
- Epoch 12 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.05% --- Training Loss: 0.9142 --- Testing Loss: 0.9010
- Epoch 13 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.05% --- Training Loss: 0.9134 --- Testing Loss: 0.9001
- Epoch 14 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.05% --- Training Loss: 0.9126 --- Testing Loss: 0.8992
- Epoch 15 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.05% --- Training Loss: 0.9119 --- Testing Loss: 0.8983
- Epoch 16 --- Training Accuracy: 55.04% --- Testing Accuracy: 56.05% --- Training Loss: 0.9111 --- Testing Loss: 0.8974
- Epoch 17 --- Training Accuracy: 55.00% --- Testing Accuracy: 55.95% --- Training Loss: 0.9103 --- Testing Loss: 0.8965
- Epoch 18 --- Training Accuracy: 55.00% --- Testing Accuracy: 55.95% --- Training Loss: 0.9095 --- Testing Loss: 0.8956
- Epoch 19 --- Training Accuracy: 54.95% --- Testing Accuracy: 55.85% --- Training Loss: 0.9088 --- Testing Loss: 0.8947
- Epoch 20 --- Training Accuracy: 54.95% --- Testing Accuracy: 55.85% --- Training Loss: 0.9080 --- Testing Loss: 0.8938
- Epoch 21 --- Training Accuracy: 54.87% --- Testing Accuracy: 55.34% --- Training Loss: 0.9073 --- Testing Loss: 0.8929
- Epoch 22 --- Training Accuracy: 54.40% --- Testing Accuracy: 54.64% --- Training Loss: 0.9066 --- Testing Loss: 0.8921
- Epoch 23 --- Training Accuracy: 53.49% --- Testing Accuracy: 54.33% --- Training Loss: 0.9059 --- Testing Loss: 0.8913
- Epoch 24 --- Training Accuracy: 53.28% --- Testing Accuracy: 54.23% --- Training Loss: 0.9052 --- Testing Loss: 0.8905
- Epoch 25 --- Training Accuracy: 53.19% --- Testing Accuracy: 54.13% --- Training Loss: 0.9046 --- Testing Loss: 0.8897
- Epoch 26 --- Training Accuracy: 53.19% --- Testing Accuracy: 54.13% --- Training Loss: 0.9039 --- Testing Loss: 0.8889
- Epoch 27 --- Training Accuracy: 53.19% --- Testing Accuracy: 54.13% --- Training Loss: 0.9033 --- Testing Loss: 0.8882
- Epoch 28 --- Training Accuracy: 53.19% --- Testing Accuracy: 54.13% --- Training Loss: 0.9026 --- Testing Loss: 0.8874
- Epoch 29 --- Training Accuracy: 53.07% --- Testing Accuracy: 54.13% --- Training Loss: 0.9018 --- Testing Loss: 0.8865
- Epoch 30 --- Training Accuracy: 53.07% --- Testing Accuracy: 54.13% --- Training Loss: 0.9011 --- Testing Loss: 0.8856

As we can see, accuracy reaches its highest value at the very first epoch and then begins to mostly decrease on the following epochs, with loss scores decreasing as well. This was expected, since as opposed to the appropriate conditions to apply this SVM loss function, there is no linearity that class A and B data present. Since our model does not quite match the margin concept of this loss function well, we do not expect it to work great with the hinge loss function.

### Log - Cosh loss
This loss function is essentially a function defined as the logarithm of the hyperbolic cosine of the prediction error. It's supposedly much smoother than the MSE loss and is not strongly affected by the occasional wildly incorrect prediciction of our model, however it's predominantly used in regression tasks. Its results for this classification task are the following: 

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.0995 --- Testing Loss: 0.1023
- Epoch 2 --- Training Accuracy: 69.92% --- Testing Accuracy: 68.75% --- Training Loss: 0.0994 --- Testing Loss: 0.1022
- Epoch 3 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.0993 --- Testing Loss: 0.1022
- Epoch 4 --- Training Accuracy: 70.05% --- Testing Accuracy: 68.95% --- Training Loss: 0.0991 --- Testing Loss: 0.1020
- Epoch 5 --- Training Accuracy: 70.99% --- Testing Accuracy: 69.65% --- Training Loss: 0.0989 --- Testing Loss: 0.1019
- Epoch 6 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 0.0988 --- Testing Loss: 0.1018
- Epoch 7 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0987 --- Testing Loss: 0.1017
- Epoch 8 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0985 --- Testing Loss: 0.1016
- Epoch 9 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0984 --- Testing Loss: 0.1015
- Epoch 10 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0983 --- Testing Loss: 0.1014
- Epoch 11 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0983 --- Testing Loss: 0.1014
- Epoch 12 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.0982 --- Testing Loss: 0.1013
- Epoch 13 --- Training Accuracy: 71.59% --- Testing Accuracy: 70.26% --- Training Loss: 0.0980 --- Testing Loss: 0.1012
- Epoch 14 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.0979 --- Testing Loss: 0.1011
- Epoch 15 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.0977 --- Testing Loss: 0.1009
- Epoch 16 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.0975 --- Testing Loss: 0.1008
- Epoch 17 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.0972 --- Testing Loss: 0.1006
- Epoch 18 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.0970 --- Testing Loss: 0.1004
- Epoch 19 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.0968 --- Testing Loss: 0.1002
- Epoch 20 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.0965 --- Testing Loss: 0.1001
- Epoch 21 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.0963 --- Testing Loss: 0.0999
- Epoch 22 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.0960 --- Testing Loss: 0.0996
- Epoch 23 --- Training Accuracy: 71.89% --- Testing Accuracy: 70.36% --- Training Loss: 0.0957 --- Testing Loss: 0.0994
- Epoch 24 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.0954 --- Testing Loss: 0.0992
- Epoch 25 --- Training Accuracy: 72.02% --- Testing Accuracy: 70.56% --- Training Loss: 0.0951 --- Testing Loss: 0.0989
- Epoch 26 --- Training Accuracy: 73.26% --- Testing Accuracy: 71.06% --- Training Loss: 0.0947 --- Testing Loss: 0.0987
- Epoch 27 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.0944 --- Testing Loss: 0.0985
- Epoch 28 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.0941 --- Testing Loss: 0.0982
- Epoch 29 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.06% --- Training Loss: 0.0939 --- Testing Loss: 0.0980
- Epoch 30 --- Training Accuracy: 73.39% --- Testing Accuracy: 71.16% --- Training Loss: 0.0936 --- Testing Loss: 0.0979

We can see that we don't reach maximum accuracy value during our training process, therefore convergence is significantly slower. In conclusion, we can see that the Nesterov Momentum Optimizer and **binary cross entropy function** combination gives us the fastest convergence possible, whereas the Nesterov Momentum Optimizer and MSE function combination gives us the minimum loss scores found, but with a slower convergence. Since loss scores are significantly lower using the MSE loss function, this is the loss function of our choosing for our future model comparisons.

## Best model yet
Upon taking into consideration both the convergence rate of each model, as well as loss score fluctuations, we consider this our best model so far:

- **Number of qubits (n_qubits)** = 4
- **Number of layers (num_layers)** = 3
- **Step size** = 0.001
- **Batch size** = 10
- **Bias term** = 0
- **Epochs** = 20
- **Optimizer** = Nesterov Momentum Optimizer
- **Loss function** = Mean Squarred Error (MSE)

Let's try changing the batch size to see if we can achieve better accuracy and/or loss scores: 

### Batch size = 50
Obviously, the only hyperparameter we changed in our best model in this instance is the batch size, which we set to a higher number, e.g. 50:

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.2086 --- Testing Loss: 0.2150
- Epoch 2 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.2084 --- Testing Loss: 0.2148
- Epoch 3 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.2081 --- Testing Loss: 0.2146
- Epoch 4 --- Training Accuracy: 70.18% --- Testing Accuracy: 69.15% --- Training Loss: 0.2076 --- Testing Loss: 0.2143
- Epoch 5 --- Training Accuracy: 71.33% --- Testing Accuracy: 70.06% --- Training Loss: 0.2071 --- Testing Loss: 0.2138
- Epoch 6 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2064 --- Testing Loss: 0.2133
- Epoch 7 --- Training Accuracy: 71.42% --- Testing Accuracy: 70.26% --- Training Loss: 0.2057 --- Testing Loss: 0.2128
- Epoch 8 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2049 --- Testing Loss: 0.2122
- Epoch 9 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2040 --- Testing Loss: 0.2116
- Epoch 10 --- Training Accuracy: 71.68% --- Testing Accuracy: 70.26% --- Training Loss: 0.2032 --- Testing Loss: 0.2109
- Epoch 11 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2023 --- Testing Loss: 0.2103
- Epoch 12 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2013 --- Testing Loss: 0.2096
- Epoch 13 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.2004 --- Testing Loss: 0.2089
- Epoch 14 --- Training Accuracy: 72.88% --- Testing Accuracy: 70.76% --- Training Loss: 0.1994 --- Testing Loss: 0.2082
- Epoch 15 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1984 --- Testing Loss: 0.2075
- Epoch 16 --- Training Accuracy: 73.35% --- Testing Accuracy: 71.06% --- Training Loss: 0.1974 --- Testing Loss: 0.2068
- Epoch 17 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1964 --- Testing Loss: 0.2061
- Epoch 18 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1954 --- Testing Loss: 0.2054
- Epoch 19 --- Training Accuracy: 73.52% --- Testing Accuracy: 71.16% --- Training Loss: 0.1944 --- Testing Loss: 0.2048
- Epoch 20 --- Training Accuracy: 78.92% --- Testing Accuracy: 75.40% --- Training Loss: 0.1933 --- Testing Loss: 0.2041

We can see that some accuracy scores have improved for both training and testing data, however the algorithm converges towards both training and testing maximum accuracy on a higher epoch. Loss scores did not get any better on any epoch. Let's try doubling it to see what results we might get:

### Batch size = 100
For double the previously tested batch size, this is what we get as a result:

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.2086 --- Testing Loss: 0.2150
- Epoch 2 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.2083 --- Testing Loss: 0.2148
- Epoch 3 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.2079 --- Testing Loss: 0.2145
- Epoch 4 --- Training Accuracy: 71.25% --- Testing Accuracy: 69.95% --- Training Loss: 0.2073 --- Testing Loss: 0.2140
- Epoch 5 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2066 --- Testing Loss: 0.2135
- Epoch 6 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2058 --- Testing Loss: 0.2129
- Epoch 7 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2049 --- Testing Loss: 0.2122
- Epoch 8 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2040 --- Testing Loss: 0.2115
- Epoch 9 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.2030 --- Testing Loss: 0.2108
- Epoch 10 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2020 --- Testing Loss: 0.2101
- Epoch 11 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.2011 --- Testing Loss: 0.2094
- Epoch 12 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.2002 --- Testing Loss: 0.2088
- Epoch 13 --- Training Accuracy: 72.92% --- Testing Accuracy: 70.96% --- Training Loss: 0.1992 --- Testing Loss: 0.2081
- Epoch 14 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1984 --- Testing Loss: 0.2075
- Epoch 15 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1975 --- Testing Loss: 0.2069
- Epoch 16 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1964 --- Testing Loss: 0.2062
- Epoch 17 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1954 --- Testing Loss: 0.2055
- Epoch 18 --- Training Accuracy: 73.52% --- Testing Accuracy: 71.16% --- Training Loss: 0.1944 --- Testing Loss: 0.2048
- Epoch 19 --- Training Accuracy: 78.88% --- Testing Accuracy: 75.40% --- Training Loss: 0.1935 --- Testing Loss: 0.2042
- Epoch 20 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1925 --- Testing Loss: 0.2036

We can see that accuracy and loss scores slightly improve, after doubling the previously used batch size.

## 2. Bloch Sphere Representation 
For this subtopic, we've created two new functions called **'get_bloch_vector'** and **'plot_bloch_sphere'** that essentially compute and plot the spheres (vectors), respectively, based on the quantum states of the quantum circuit. More implementation specific code details are included in provided notebook file.
What we will mention, however, is the following: 
- **Convention**: We plot the bloch sphere representation for the first qubit of the 6 randomly chosen points (in total) of the last set of 5 epochs. Each one of these 6 randomly selected points corresponds to a quantum state, where each quantum state is represented by 4 qubits. So, we plot 6 bloch spheres in total, in order to justify our model and plot respective quantum states. This is done, because we want to monitor our model's performance to see how it "responds" to randomly picked data points. Also, plotting every qubit of these states gives us the exact same plot (for every state), so that's why we only choose the first qubit.
- **Justification**: In plotted bloch spheres, we can see how our Variational Quantum Classifier (VQC) separates and transforms the states using the respective vectors. Each arrow represents the state of a qubit after the VQC has processed this randomly picked data point. The **direction** of the arrows represents the quantum state of the qubit, where different directions indicate different superpositions of basis states ∣0⟩ and ∣1⟩. These arrows point to different directions for different input data points, which means that the model is responding well to different inputs, as it is learning to differentiate between classes A and B. This means that, ideally for this binary classification task, the states corresponding to different classes should be distinguishable in our Bloch sphere plots. This indicates some **consistency** to our model as well, since the quantum states of the qubits are being consistently transformed by the variational circuit. That is, generally, a good sign, since the circuit is possibly learning a pattern in our data. In conclusion, our VQC is consistently learning some representation of input data.

## 3. Classical Neural Network (NN) that solves the parity problem
If we want to solve this parity problem using classical Machine Learning techniques, we can always experiment with Neural Networks. We will be using package 'keras', as it is the most commonly used Neural Network API in python. In order to proceed to accurate comparisons, we run our NN model using *either* the Adam optimizer with the binary cross entropy loss function (which is considered the best optimization and loss function combination for such binary classification tasks) *or* the Nesterov Momentum optimizer with the MSE loss function. Conventionally, we will experiment with the numbers of epochs previously used (20 and 30), as well as batch sizes of 10 and 5, keeping the learning rate fixed to 0.001. We will, now, report results that came from the following models:

### Nesterov Momentum optimizer, MSE

**For 20 epochs and batch size of 10:**

- Epoch 1/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.7816 - loss: 0.2370 - val_accuracy: 0.7540 - val_loss: 0.2021
- Epoch 2/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8353 - loss: 0.1896 - val_accuracy: 0.7540 - val_loss: 0.1754
- Epoch 3/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8181 - loss: 0.1626 - val_accuracy: 0.7540 - val_loss: 0.1616
- Epoch 4/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8095 - loss: 0.1522 - val_accuracy: 0.7540 - val_loss: 0.1519
- Epoch 5/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8063 - loss: 0.1420 - val_accuracy: 0.7560 - val_loss: 0.1440
- Epoch 6/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8130 - loss: 0.1322 - val_accuracy: 0.7560 - val_loss: 0.1362
- Epoch 7/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8158 - loss: 0.1298 - val_accuracy: 0.7421 - val_loss: 0.1281
- Epoch 8/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8135 - loss: 0.1226 - val_accuracy: 0.7321 - val_loss: 0.1201
- Epoch 9/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8381 - loss: 0.1167 - val_accuracy: 0.7500 - val_loss: 0.1123
- Epoch 10/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8589 - loss: 0.1113 - val_accuracy: 0.8075 - val_loss: 0.1053
- Epoch 11/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8510 - loss: 0.1053 - val_accuracy: 0.8413 - val_loss: 0.0982
- Epoch 12/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8903 - loss: 0.0989 - val_accuracy: 0.9187 - val_loss: 0.0915
- Epoch 13/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8837 - loss: 0.0950 - val_accuracy: 0.9683 - val_loss: 0.0853
- Epoch 14/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9067 - loss: 0.0880 - val_accuracy: 0.9742 - val_loss: 0.0794
- Epoch 15/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9076 - loss: 0.0888 - val_accuracy: 0.9742 - val_loss: 0.0742
- Epoch 16/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9120 - loss: 0.0814 - val_accuracy: 0.9742 - val_loss: 0.0693
- Epoch 17/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9439 - loss: 0.0737 - val_accuracy: 0.9742 - val_loss: 0.0649
- Epoch 18/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9340 - loss: 0.0739 - val_accuracy: 0.9742 - val_loss: 0.0611
- Epoch 19/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9495 - loss: 0.0626 - val_accuracy: 0.9742 - val_loss: 0.0576
- Epoch 20/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9428 - loss: 0.0628 - val_accuracy: 0.9742 - val_loss: 0.0547
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9699 - loss: 0.0564 
- Testing Accuracy: 97.42%  --- Testing Loss: 0.0547

We can already see that both accuracy and loss function scores are much better than before. Since results are looking fruitful, let's try increasing the number of epochs used.

For 30 epochs and batch size of 10 :

-  Epoch 1/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7982 - loss: 0.2335 - val_accuracy: 0.7421 - val_loss: 0.2005
-  Epoch 2/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8300 - loss: 0.1882 - val_accuracy: 0.7540 - val_loss: 0.1760
-  Epoch 3/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8186 - loss: 0.1607 - val_accuracy: 0.7540 - val_loss: 0.1634
-  Epoch 4/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8119 - loss: 0.1520 - val_accuracy: 0.7540 - val_loss: 0.1556
-  Epoch 5/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8035 - loss: 0.1464 - val_accuracy: 0.7540 - val_loss: 0.1493
-  Epoch 6/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8216 - loss: 0.1369 - val_accuracy: 0.7540 - val_loss: 0.1435
-  Epoch 7/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8119 - loss: 0.1316 - val_accuracy: 0.7421 - val_loss: 0.1374
-  Epoch 8/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8196 - loss: 0.1277 - val_accuracy: 0.7440 - val_loss: 0.1319
-  Epoch 9/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8279 - loss: 0.1216 - val_accuracy: 0.7440 - val_loss: 0.1261
-  Epoch 10/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8431 - loss: 0.1145 - val_accuracy: 0.7480 - val_loss: 0.1199
-  Epoch 11/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8523 - loss: 0.1075 - val_accuracy: 0.7917 - val_loss: 0.1132
-  Epoch 12/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8551 - loss: 0.1086 - val_accuracy: 0.8254 - val_loss: 0.1071
-  Epoch 13/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8739 - loss: 0.0984 - val_accuracy: 0.8869 - val_loss: 0.1008
-  Epoch 14/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8509 - loss: 0.1057 - val_accuracy: 0.9048 - val_loss: 0.0953
-  Epoch 15/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8660 - loss: 0.1006 - val_accuracy: 0.9048 - val_loss: 0.0898
-  Epoch 16/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8767 - loss: 0.0918 - val_accuracy: 0.9048 - val_loss: 0.0846
-  Epoch 17/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9071 - loss: 0.0856 - val_accuracy: 0.9087 - val_loss: 0.0796
-  Epoch 18/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9066 - loss: 0.0838 - val_accuracy: 0.9286 - val_loss: 0.0752
-  Epoch 19/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8963 - loss: 0.0838 - val_accuracy: 0.9742 - val_loss: 0.0712
-  Epoch 20/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9318 - loss: 0.0704 - val_accuracy: 0.9742 - val_loss: 0.0674
-  Epoch 21/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9174 - loss: 0.0743 - val_accuracy: 0.9742 - val_loss: 0.0637
-  Epoch 22/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9347 - loss: 0.0671 - val_accuracy: 0.9742 - val_loss: 0.0605
-  Epoch 23/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9293 - loss: 0.0690 - val_accuracy: 0.9742 - val_loss: 0.0575
-  Epoch 24/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9249 - loss: 0.0683 - val_accuracy: 0.9742 - val_loss: 0.0549
-  Epoch 25/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9307 - loss: 0.0686 - val_accuracy: 0.9742 - val_loss: 0.0522
-  Epoch 26/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9462 - loss: 0.0606 - val_accuracy: 0.9742 - val_loss: 0.0500
-  Epoch 27/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9575 - loss: 0.0558 - val_accuracy: 0.9742 - val_loss: 0.0480
-  Epoch 28/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9516 - loss: 0.0539 - val_accuracy: 0.9742 - val_loss: 0.0462
-  Epoch 29/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9418 - loss: 0.0541 - val_accuracy: 0.9742 - val_loss: 0.0447
-  Epoch 30/30
-  136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9513 - loss: 0.0501 - val_accuracy: 0.9742 - val_loss: 0.0433
-  16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9699 - loss: 0.0460 
-  Testing Accuracy: 97.42%  --- Testing Loss: 0.0433

Our final testing accuracy remains the same, but we can stil see the consistent improvements in our final loss score. 

**For 20 epochs and batch size of 5:**

- Epoch 1/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.8007 - loss: 0.2239 - val_accuracy: 0.7540 - val_loss: 0.1708
- Epoch 2/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8105 - loss: 0.1585 - val_accuracy: 0.7540 - val_loss: 0.1498
- Epoch 3/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8032 - loss: 0.1379 - val_accuracy: 0.7560 - val_loss: 0.1334
- Epoch 4/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8090 - loss: 0.1257 - val_accuracy: 0.7421 - val_loss: 0.1163
- Epoch 5/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8352 - loss: 0.1127 - val_accuracy: 0.8413 - val_loss: 0.0989
- Epoch 6/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8781 - loss: 0.0958 - val_accuracy: 0.9742 - val_loss: 0.0831
- Epoch 7/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8961 - loss: 0.0898 - val_accuracy: 0.9742 - val_loss: 0.0701
- Epoch 8/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9145 - loss: 0.0772 - val_accuracy: 0.9742 - val_loss: 0.0605
- Epoch 9/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9255 - loss: 0.0702 - val_accuracy: 0.9742 - val_loss: 0.0530
- Epoch 10/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9459 - loss: 0.0592 - val_accuracy: 0.9742 - val_loss: 0.0475
- Epoch 11/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9491 - loss: 0.0565 - val_accuracy: 0.9742 - val_loss: 0.0431
- Epoch 12/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9595 - loss: 0.0521 - val_accuracy: 0.9742 - val_loss: 0.0399
- Epoch 13/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9591 - loss: 0.0491 - val_accuracy: 0.9742 - val_loss: 0.0374
- Epoch 14/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9654 - loss: 0.0438 - val_accuracy: 0.9742 - val_loss: 0.0355
- Epoch 15/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9600 - loss: 0.0421 - val_accuracy: 0.9742 - val_loss: 0.0340
- Epoch 16/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9659 - loss: 0.0368 - val_accuracy: 0.9742 - val_loss: 0.0328
- Epoch 17/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9785 - loss: 0.0355 - val_accuracy: 0.9742 - val_loss: 0.0317
- Epoch 18/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9685 - loss: 0.0347 - val_accuracy: 0.9742 - val_loss: 0.0310
- Epoch 19/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9757 - loss: 0.0347 - val_accuracy: 0.9742 - val_loss: 0.0304
- Epoch 20/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9792 - loss: 0.0318 - val_accuracy: 0.9742 - val_loss: 0.0296
- Epoch 21/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9674 - loss: 0.0336 - val_accuracy: 0.9742 - val_loss: 0.0293
- Epoch 22/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9721 - loss: 0.0311 - val_accuracy: 0.9742 - val_loss: 0.0288
- Epoch 23/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9682 - loss: 0.0317 - val_accuracy: 0.9742 - val_loss: 0.0285
- Epoch 24/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9754 - loss: 0.0303 - val_accuracy: 0.9742 - val_loss: 0.0282
- Epoch 25/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9795 - loss: 0.0272 - val_accuracy: 0.9742 - val_loss: 0.0280
- Epoch 26/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9758 - loss: 0.0299 - val_accuracy: 0.9742 - val_loss: 0.0278
- Epoch 27/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9756 - loss: 0.0274 - val_accuracy: 0.9742 - val_loss: 0.0277
- Epoch 28/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9803 - loss: 0.0249 - val_accuracy: 0.9742 - val_loss: 0.0275
- Epoch 29/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9791 - loss: 0.0266 - val_accuracy: 0.9742 - val_loss: 0.0274
- Epoch 30/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9763 - loss: 0.0242 - val_accuracy: 0.9742 - val_loss: 0.0272
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9699 - loss: 0.0313 
- Testing Accuracy: 97.42%  --- Testing Loss: 0.0272

Although testing accuracy remains the same, testing loss has significantly improved. Let's try increasing the number of epochs for this batch size.

**For 30 epochs and batch size of 5:**

- Epoch 1/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.8285 - loss: 0.2250 - val_accuracy: 0.7540 - val_loss: 0.1784
- Epoch 2/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.7995 - loss: 0.1610 - val_accuracy: 0.7540 - val_loss: 0.1564
- Epoch 3/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8000 - loss: 0.1420 - val_accuracy: 0.7540 - val_loss: 0.1428
- Epoch 4/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8042 - loss: 0.1304 - val_accuracy: 0.7440 - val_loss: 0.1295
- Epoch 5/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8113 - loss: 0.1227 - val_accuracy: 0.7321 - val_loss: 0.1151
- Epoch 6/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8251 - loss: 0.1146 - val_accuracy: 0.7520 - val_loss: 0.1012
- Epoch 7/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8697 - loss: 0.1013 - val_accuracy: 0.9087 - val_loss: 0.0885
- Epoch 8/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8798 - loss: 0.0902 - val_accuracy: 0.9742 - val_loss: 0.0769
- Epoch 9/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9093 - loss: 0.0798 - val_accuracy: 0.9742 - val_loss: 0.0672
- Epoch 10/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9164 - loss: 0.0754 - val_accuracy: 0.9742 - val_loss: 0.0593
- Epoch 11/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9258 - loss: 0.0690 - val_accuracy: 0.9742 - val_loss: 0.0534
- Epoch 12/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9407 - loss: 0.0643 - val_accuracy: 0.9742 - val_loss: 0.0484
- Epoch 13/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9428 - loss: 0.0612 - val_accuracy: 0.9742 - val_loss: 0.0445
- Epoch 14/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9539 - loss: 0.0544 - val_accuracy: 0.9742 - val_loss: 0.0416
- Epoch 15/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9657 - loss: 0.0473 - val_accuracy: 0.9742 - val_loss: 0.0390
- Epoch 16/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9593 - loss: 0.0485 - val_accuracy: 0.9742 - val_loss: 0.0371
- Epoch 17/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9682 - loss: 0.0448 - val_accuracy: 0.9742 - val_loss: 0.0353
- Epoch 18/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9717 - loss: 0.0422 - val_accuracy: 0.9742 - val_loss: 0.0339
- Epoch 19/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9679 - loss: 0.0389 - val_accuracy: 0.9742 - val_loss: 0.0328
- Epoch 20/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9652 - loss: 0.0405 - val_accuracy: 0.9742 - val_loss: 0.0318
- Epoch 21/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9745 - loss: 0.0352 - val_accuracy: 0.9742 - val_loss: 0.0310
- Epoch 22/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9715 - loss: 0.0321 - val_accuracy: 0.9742 - val_loss: 0.0303
- Epoch 23/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9707 - loss: 0.0336 - val_accuracy: 0.9742 - val_loss: 0.0298
- Epoch 24/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9805 - loss: 0.0280 - val_accuracy: 0.9742 - val_loss: 0.0294
- Epoch 25/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9801 - loss: 0.0303 - val_accuracy: 0.9742 - val_loss: 0.0291
- Epoch 26/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9816 - loss: 0.0292 - val_accuracy: 0.9742 - val_loss: 0.0286
- Epoch 27/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9778 - loss: 0.0288 - val_accuracy: 0.9742 - val_loss: 0.0283
- Epoch 28/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9757 - loss: 0.0321 - val_accuracy: 0.9742 - val_loss: 0.0280
- Epoch 29/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9782 - loss: 0.0300 - val_accuracy: 0.9742 - val_loss: 0.0278
- Epoch 30/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9757 - loss: 0.0297 - val_accuracy: 0.9742 - val_loss: 0.0276
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9699 - loss: 0.0318 
- Testing Accuracy: 97.42%  --- Testing Loss: 0.0276

Here, although accuracy scores remain the same, testing loss seems to be a little bit higher than before. After multiple executions, we determine that loss scores remain a little bit higher on every epoch. Still, we're talking about marginal differences.

### Adam Optimizer, Binary Cross Entropy

**For 20 epochs and batch size of 10:**

- Epoch 1/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.8355 - loss: 0.6204 - val_accuracy: 0.9742 - val_loss: 0.3289
- Epoch 2/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9453 - loss: 0.2774 - val_accuracy: 0.9742 - val_loss: 0.1476
- Epoch 3/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9753 - loss: 0.1388 - val_accuracy: 0.9742 - val_loss: 0.1168
- Epoch 4/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9807 - loss: 0.1077 - val_accuracy: 0.9742 - val_loss: 0.1032
- Epoch 5/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9849 - loss: 0.0954 - val_accuracy: 0.9742 - val_loss: 0.0937
- Epoch 6/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9854 - loss: 0.0803 - val_accuracy: 0.9742 - val_loss: 0.0859
- Epoch 7/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9836 - loss: 0.0625 - val_accuracy: 0.9742 - val_loss: 0.0803
- Epoch 8/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9851 - loss: 0.0598 - val_accuracy: 0.9742 - val_loss: 0.0758
- Epoch 9/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9843 - loss: 0.0543 - val_accuracy: 0.9742 - val_loss: 0.0688
- Epoch 10/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9855 - loss: 0.0511 - val_accuracy: 0.9742 - val_loss: 0.0622
- Epoch 11/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9824 - loss: 0.0472 - val_accuracy: 0.9742 - val_loss: 0.0576
- Epoch 12/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0485 - val_accuracy: 0.9742 - val_loss: 0.0505
- Epoch 13/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9852 - loss: 0.0487 - val_accuracy: 0.9742 - val_loss: 0.0444
- Epoch 14/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9859 - loss: 0.0365 - val_accuracy: 0.9742 - val_loss: 0.0424
- Epoch 15/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9872 - loss: 0.0297 - val_accuracy: 0.9742 - val_loss: 0.0372
- Epoch 16/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9874 - loss: 0.0324 - val_accuracy: 0.9742 - val_loss: 0.0324
- Epoch 17/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9894 - loss: 0.0282 - val_accuracy: 0.9881 - val_loss: 0.0265
- Epoch 18/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9833 - loss: 0.0299 - val_accuracy: 0.9782 - val_loss: 0.0252
- Epoch 19/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9912 - loss: 0.0242 - val_accuracy: 0.9881 - val_loss: 0.0216
- Epoch 20/20
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9933 - loss: 0.0212 - val_accuracy: 0.9881 - val_loss: 0.0198
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 933us/step - accuracy: 0.9891 - loss: 0.0222
- Testing Accuracy: 98.81%  --- Testing Loss: 0.0198

Although results may vary, because in some executions we get a testing accuracy of a value between 97.42 and 98.81 %, this model is generally more efficient than our previous one. We notice that, as opposed to our quantum circuit, this generally optimal combination gives us better results than the previous model we tested. Let's try slightly increasing number of epochs used.

**For 30 epochs and batch size of 10:**

- Epoch 1/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.8404 - loss: 0.6270 - val_accuracy: 0.9742 - val_loss: 0.3010
- Epoch 2/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9743 - loss: 0.2186 - val_accuracy: 0.9742 - val_loss: 0.1291
- Epoch 3/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9862 - loss: 0.0979 - val_accuracy: 0.9742 - val_loss: 0.1050
- Epoch 4/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9857 - loss: 0.0769 - val_accuracy: 0.9742 - val_loss: 0.0901
- Epoch 5/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9854 - loss: 0.0623 - val_accuracy: 0.9742 - val_loss: 0.0801
- Epoch 6/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9858 - loss: 0.0498 - val_accuracy: 0.9742 - val_loss: 0.0689
- Epoch 7/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9862 - loss: 0.0405 - val_accuracy: 0.9762 - val_loss: 0.0566
- Epoch 8/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9859 - loss: 0.0367 - val_accuracy: 0.9762 - val_loss: 0.0459
- Epoch 9/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9935 - loss: 0.0267 - val_accuracy: 0.9762 - val_loss: 0.0360
- Epoch 10/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9941 - loss: 0.0216 - val_accuracy: 0.9762 - val_loss: 0.0310
- Epoch 11/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9961 - loss: 0.0192 - val_accuracy: 0.9782 - val_loss: 0.0238
- Epoch 12/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9941 - loss: 0.0161 - val_accuracy: 0.9921 - val_loss: 0.0197
- Epoch 13/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9982 - loss: 0.0117 - val_accuracy: 0.9980 - val_loss: 0.0165
- Epoch 14/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9924 - loss: 0.0184 - val_accuracy: 0.9980 - val_loss: 0.0141
- Epoch 15/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9990 - loss: 0.0096 - val_accuracy: 0.9980 - val_loss: 0.0126
- Epoch 16/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9989 - loss: 0.0093 - val_accuracy: 0.9980 - val_loss: 0.0109
- Epoch 17/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0075 - val_accuracy: 0.9980 - val_loss: 0.0106
- Epoch 18/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9986 - loss: 0.0072 - val_accuracy: 0.9980 - val_loss: 0.0093
- Epoch 19/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9984 - loss: 0.0061 - val_accuracy: 0.9980 - val_loss: 0.0087
- Epoch 20/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9994 - loss: 0.0058 - val_accuracy: 0.9980 - val_loss: 0.0089
- Epoch 21/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0049 - val_accuracy: 0.9980 - val_loss: 0.0076
- Epoch 22/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9962 - loss: 0.0067 - val_accuracy: 0.9980 - val_loss: 0.0073
- Epoch 23/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9995 - loss: 0.0054 - val_accuracy: 0.9980 - val_loss: 0.0070
- Epoch 24/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9986 - loss: 0.0052 - val_accuracy: 0.9980 - val_loss: 0.0067
- Epoch 25/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0041 - val_accuracy: 0.9980 - val_loss: 0.0067
- Epoch 26/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0040 - val_accuracy: 0.9980 - val_loss: 0.0065
- Epoch 27/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0047 - val_accuracy: 0.9980 - val_loss: 0.0063
- Epoch 28/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9973 - loss: 0.0043 - val_accuracy: 0.9980 - val_loss: 0.0063
- Epoch 29/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9997 - loss: 0.0036 - val_accuracy: 0.9980 - val_loss: 0.0062
- Epoch 30/30
- 136/136 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9997 - loss: 0.0032 - val_accuracy: 0.9980 - val_loss: 0.0060
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9982 - loss: 0.0069 
- Testing Accuracy: 99.80%  --- Testing Loss: 0.0060

After multiple executions, this is the best result we seem to get. Accuracy scores are as close to 100% as possible, whereas final loss function score has reached new minimum value.

**For 20 epochs and batch size of 5:**

- Epoch 1/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.8281 - loss: 0.6070 - val_accuracy: 0.9742 - val_loss: 0.2505
- Epoch 2/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9672 - loss: 0.2063 - val_accuracy: 0.9742 - val_loss: 0.1220
- Epoch 3/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9818 - loss: 0.1078 - val_accuracy: 0.9742 - val_loss: 0.0991
- Epoch 4/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9817 - loss: 0.0843 - val_accuracy: 0.9742 - val_loss: 0.0893
- Epoch 5/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9852 - loss: 0.0610 - val_accuracy: 0.9742 - val_loss: 0.0807
- Epoch 6/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9857 - loss: 0.0545 - val_accuracy: 0.9742 - val_loss: 0.0700
- Epoch 7/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9823 - loss: 0.0543 - val_accuracy: 0.9742 - val_loss: 0.0650
- Epoch 8/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9893 - loss: 0.0399 - val_accuracy: 0.9742 - val_loss: 0.0571
- Epoch 9/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9848 - loss: 0.0405 - val_accuracy: 0.9762 - val_loss: 0.0481
- Epoch 10/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9884 - loss: 0.0397 - val_accuracy: 0.9762 - val_loss: 0.0421
- Epoch 11/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9842 - loss: 0.0336 - val_accuracy: 0.9762 - val_loss: 0.0371
- Epoch 12/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9904 - loss: 0.0250 - val_accuracy: 0.9762 - val_loss: 0.0376
- Epoch 13/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9878 - loss: 0.0341 - val_accuracy: 0.9861 - val_loss: 0.0262
- Epoch 14/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9918 - loss: 0.0319 - val_accuracy: 0.9861 - val_loss: 0.0252
- Epoch 15/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9910 - loss: 0.0255 - val_accuracy: 0.9861 - val_loss: 0.0210
- Epoch 16/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9966 - loss: 0.0193 - val_accuracy: 0.9881 - val_loss: 0.0176
- Epoch 17/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9969 - loss: 0.0200 - val_accuracy: 0.9980 - val_loss: 0.0140
- Epoch 18/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9947 - loss: 0.0140 - val_accuracy: 0.9980 - val_loss: 0.0123
- Epoch 19/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9938 - loss: 0.0236 - val_accuracy: 0.9980 - val_loss: 0.0144
- Epoch 20/20
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9961 - loss: 0.0171 - val_accuracy: 0.9980 - val_loss: 0.0113
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 934us/step - accuracy: 0.9982 - loss: 0.0118
- Testing Accuracy: 99.80%  --- Testing Loss: 0.0113

**For 30 epochs and batch size of 5:**

- Epoch 1/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.8593 - loss: 0.5636 - val_accuracy: 0.9742 - val_loss: 0.1895
- Epoch 2/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9639 - loss: 0.1777 - val_accuracy: 0.9742 - val_loss: 0.1188
- Epoch 3/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9807 - loss: 0.0997 - val_accuracy: 0.9742 - val_loss: 0.1031
- Epoch 4/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9841 - loss: 0.0743 - val_accuracy: 0.9742 - val_loss: 0.0979
- Epoch 5/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9850 - loss: 0.0663 - val_accuracy: 0.9742 - val_loss: 0.0946
- Epoch 6/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9835 - loss: 0.0580 - val_accuracy: 0.9742 - val_loss: 0.0849
- Epoch 7/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9842 - loss: 0.0570 - val_accuracy: 0.9762 - val_loss: 0.0761
- Epoch 8/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9856 - loss: 0.0484 - val_accuracy: 0.9762 - val_loss: 0.0676
- Epoch 9/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9889 - loss: 0.0419 - val_accuracy: 0.9762 - val_loss: 0.0674
- Epoch 10/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9904 - loss: 0.0370 - val_accuracy: 0.9762 - val_loss: 0.0622
- Epoch 11/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9877 - loss: 0.0331 - val_accuracy: 0.9762 - val_loss: 0.0552
- Epoch 12/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9907 - loss: 0.0454 - val_accuracy: 0.9762 - val_loss: 0.0451
- Epoch 13/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9924 - loss: 0.0252 - val_accuracy: 0.9762 - val_loss: 0.0430
- Epoch 14/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9913 - loss: 0.0258 - val_accuracy: 0.9742 - val_loss: 0.0358
- Epoch 15/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9927 - loss: 0.0306 - val_accuracy: 0.9742 - val_loss: 0.0303
- Epoch 16/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9934 - loss: 0.0252 - val_accuracy: 0.9742 - val_loss: 0.0271
- Epoch 17/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9947 - loss: 0.0184 - val_accuracy: 0.9742 - val_loss: 0.0269
- Epoch 18/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9955 - loss: 0.0163 - val_accuracy: 0.9742 - val_loss: 0.0256
- Epoch 19/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9890 - loss: 0.0242 - val_accuracy: 0.9881 - val_loss: 0.0206
- Epoch 20/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9933 - loss: 0.0182 - val_accuracy: 0.9881 - val_loss: 0.0178
- Epoch 21/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9953 - loss: 0.0213 - val_accuracy: 0.9921 - val_loss: 0.0157
- Epoch 22/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9947 - loss: 0.0179 - val_accuracy: 0.9980 - val_loss: 0.0160
- Epoch 23/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9940 - loss: 0.0137 - val_accuracy: 0.9881 - val_loss: 0.0161
- Epoch 24/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9956 - loss: 0.0146 - val_accuracy: 0.9980 - val_loss: 0.0147
- Epoch 25/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9938 - loss: 0.0162 - val_accuracy: 0.9980 - val_loss: 0.0137
- Epoch 26/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9956 - loss: 0.0127 - val_accuracy: 0.9980 - val_loss: 0.0131
- Epoch 27/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9940 - loss: 0.0173 - val_accuracy: 0.9980 - val_loss: 0.0132
- Epoch 28/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9983 - loss: 0.0091 - val_accuracy: 0.9980 - val_loss: 0.0130
- Epoch 29/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9978 - loss: 0.0098 - val_accuracy: 0.9980 - val_loss: 0.0123
- Epoch 30/30
- 271/271 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9934 - loss: 0.0147 - val_accuracy: 0.9980 - val_loss: 0.0143
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 800us/step - accuracy: 0.9982 - loss: 0.0164
- Testing Accuracy: 99.80%  --- Testing Loss: 0.0143

Let's try keeping number of epochs fixed and increasing / decreasing the batch size (perhaps to some values like the ones we set before):

**For 30 epochs and batch size of 2:**

- Epoch 1/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.8579 - loss: 0.5058 - val_accuracy: 0.9742 - val_loss: 0.1440
- Epoch 2/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9790 - loss: 0.1231 - val_accuracy: 0.9742 - val_loss: 0.0985
- Epoch 3/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9801 - loss: 0.0776 - val_accuracy: 0.9742 - val_loss: 0.0874
- Epoch 4/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9779 - loss: 0.0681 - val_accuracy: 0.9742 - val_loss: 0.0732
- Epoch 5/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9898 - loss: 0.0439 - val_accuracy: 0.9742 - val_loss: 0.0657
- Epoch 6/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9862 - loss: 0.0431 - val_accuracy: 0.9762 - val_loss: 0.0582
- Epoch 7/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9922 - loss: 0.0303 - val_accuracy: 0.9762 - val_loss: 0.0477
- Epoch 8/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9931 - loss: 0.0284 - val_accuracy: 0.9762 - val_loss: 0.0397
- Epoch 9/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9929 - loss: 0.0282 - val_accuracy: 0.9762 - val_loss: 0.0317
- Epoch 10/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9921 - loss: 0.0204 - val_accuracy: 0.9742 - val_loss: 0.0305
- Epoch 11/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9955 - loss: 0.0195 - val_accuracy: 0.9881 - val_loss: 0.0253
- Epoch 12/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9917 - loss: 0.0187 - val_accuracy: 0.9861 - val_loss: 0.0202
- Epoch 13/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9924 - loss: 0.0211 - val_accuracy: 0.9861 - val_loss: 0.0180
- Epoch 14/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9946 - loss: 0.0137 - val_accuracy: 0.9881 - val_loss: 0.0161
- Epoch 15/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9950 - loss: 0.0153 - val_accuracy: 0.9861 - val_loss: 0.0176
- Epoch 16/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9961 - loss: 0.0137 - val_accuracy: 0.9881 - val_loss: 0.0142
- Epoch 17/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9971 - loss: 0.0088 - val_accuracy: 0.9980 - val_loss: 0.0132
- Epoch 18/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9969 - loss: 0.0093 - val_accuracy: 0.9980 - val_loss: 0.0134
- Epoch 19/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9927 - loss: 0.0117 - val_accuracy: 0.9980 - val_loss: 0.0140
- Epoch 20/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9921 - loss: 0.0126 - val_accuracy: 0.9980 - val_loss: 0.0121
- Epoch 21/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9981 - loss: 0.0091 - val_accuracy: 0.9980 - val_loss: 0.0122
- Epoch 22/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9963 - loss: 0.0087 - val_accuracy: 0.9980 - val_loss: 0.0114
- Epoch 23/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9954 - loss: 0.0150 - val_accuracy: 0.9980 - val_loss: 0.0115
- Epoch 24/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9992 - loss: 0.0057 - val_accuracy: 0.9980 - val_loss: 0.0107
- Epoch 25/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9978 - loss: 0.0074 - val_accuracy: 0.9980 - val_loss: 0.0100
- Epoch 26/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9957 - loss: 0.0083 - val_accuracy: 0.9980 - val_loss: 0.0099
- Epoch 27/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9964 - loss: 0.0102 - val_accuracy: 0.9980 - val_loss: 0.0098
- Epoch 28/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9990 - loss: 0.0052 - val_accuracy: 0.9980 - val_loss: 0.0096
- Epoch 29/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9980 - loss: 0.0056 - val_accuracy: 0.9980 - val_loss: 0.0092
- Epoch 30/30
- 677/677 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9990 - loss: 0.0041 - val_accuracy: 0.9980 - val_loss: 0.0091
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 867us/step - accuracy: 0.9982 - loss: 0.0091
- Testing Accuracy: 99.80%  --- Testing Loss: 0.0091

**For 30 epochs and batch size of 50:**

- Epoch 1/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.7836 - loss: 0.6843 - val_accuracy: 0.8948 - val_loss: 0.6193
- Epoch 2/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9030 - loss: 0.6090 - val_accuracy: 0.9742 - val_loss: 0.5310
- Epoch 3/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9313 - loss: 0.5085 - val_accuracy: 0.9742 - val_loss: 0.4120
- Epoch 4/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9232 - loss: 0.3969 - val_accuracy: 0.9742 - val_loss: 0.2882
- Epoch 5/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9510 - loss: 0.2978 - val_accuracy: 0.9742 - val_loss: 0.2049
- Epoch 6/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9528 - loss: 0.2247 - val_accuracy: 0.9742 - val_loss: 0.1583
- Epoch 7/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9711 - loss: 0.1769 - val_accuracy: 0.9742 - val_loss: 0.1343
- Epoch 8/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9687 - loss: 0.1556 - val_accuracy: 0.9742 - val_loss: 0.1195
- Epoch 9/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9748 - loss: 0.1312 - val_accuracy: 0.9742 - val_loss: 0.1132
- Epoch 10/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9794 - loss: 0.1093 - val_accuracy: 0.9742 - val_loss: 0.1060
- Epoch 11/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9784 - loss: 0.1090 - val_accuracy: 0.9742 - val_loss: 0.1004
- Epoch 12/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9756 - loss: 0.0999 - val_accuracy: 0.9742 - val_loss: 0.0965
- Epoch 13/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9803 - loss: 0.1055 - val_accuracy: 0.9742 - val_loss: 0.0930
- Epoch 14/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9791 - loss: 0.0871 - val_accuracy: 0.9742 - val_loss: 0.0883
- Epoch 15/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9828 - loss: 0.0831 - val_accuracy: 0.9742 - val_loss: 0.0834
- Epoch 16/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9798 - loss: 0.0748 - val_accuracy: 0.9742 - val_loss: 0.0798
- Epoch 17/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9830 - loss: 0.0745 - val_accuracy: 0.9742 - val_loss: 0.0765
- Epoch 18/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9799 - loss: 0.0848 - val_accuracy: 0.9742 - val_loss: 0.0741
- Epoch 19/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9833 - loss: 0.0661 - val_accuracy: 0.9742 - val_loss: 0.0725
- Epoch 20/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9882 - loss: 0.0607 - val_accuracy: 0.9742 - val_loss: 0.0697
- Epoch 21/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9834 - loss: 0.0629 - val_accuracy: 0.9742 - val_loss: 0.0670
- Epoch 22/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9858 - loss: 0.0592 - val_accuracy: 0.9742 - val_loss: 0.0650
- Epoch 23/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9845 - loss: 0.0541 - val_accuracy: 0.9742 - val_loss: 0.0635
- Epoch 24/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9866 - loss: 0.0480 - val_accuracy: 0.9742 - val_loss: 0.0609
- Epoch 25/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9816 - loss: 0.0539 - val_accuracy: 0.9742 - val_loss: 0.0586
- Epoch 26/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9833 - loss: 0.0547 - val_accuracy: 0.9742 - val_loss: 0.0578
- Epoch 27/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9866 - loss: 0.0496 - val_accuracy: 0.9742 - val_loss: 0.0551
- Epoch 28/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9869 - loss: 0.0448 - val_accuracy: 0.9742 - val_loss: 0.0533
- Epoch 29/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9860 - loss: 0.0467 - val_accuracy: 0.9742 - val_loss: 0.0510
- Epoch 30/30
- 28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9888 - loss: 0.0388 - val_accuracy: 0.9742 - val_loss: 0.0498
- 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 867us/step - accuracy: 0.9699 - loss: 0.0623
- Testing Accuracy: 97.42%  --- Testing Loss: 0.0498

In conclusion, the "batch size" hyperparameter, alongside the optimization and loss functions previously tested, play a significant role in the determination of our accuracy and loss scores. We can see that our classical Neural Network (NN) model generally performs better than our VQC one, which was to some degree expected.


## 4. Parameter Shift Rule
In order to include this  rule in our measurements, we only need to modify PennyLane's quantum node (qnode) definition to include the 'diff_method' argument as such:

    @qml.qnode(dev, interface = 'autograd', diff_method = 'parameter-shift')

This line of code is, by default, commented out, but it can be manually commented in by the user. If it's commented in, make sure to also change the **'include_parameter_shift'** argument to **True**, instead of False, in code cell where function **'train_VQC'** is ran.

### Results
For 30 epochs (and hyperparameters of our best model), these are the quantum circuit results using the parameter shift method:

- Epoch 1 --- Training Accuracy: 69.88% --- Testing Accuracy: 68.75% --- Training Loss: 0.2084 --- Testing Loss: 0.2148
- Epoch 2 --- Training Accuracy: 69.96% --- Testing Accuracy: 68.75% --- Training Loss: 0.2079 --- Testing Loss: 0.2145
- Epoch 3 --- Training Accuracy: 70.99% --- Testing Accuracy: 69.65% --- Training Loss: 0.2074 --- Testing Loss: 0.2141
- Epoch 4 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2066 --- Testing Loss: 0.2135
- Epoch 5 --- Training Accuracy: 71.38% --- Testing Accuracy: 70.16% --- Training Loss: 0.2061 --- Testing Loss: 0.2131
- Epoch 6 --- Training Accuracy: 71.51% --- Testing Accuracy: 70.26% --- Training Loss: 0.2056 --- Testing Loss: 0.2127
- Epoch 7 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2050 --- Testing Loss: 0.2123
- Epoch 8 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2045 --- Testing Loss: 0.2120
- Epoch 9 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2043 --- Testing Loss: 0.2118
- Epoch 10 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2040 --- Testing Loss: 0.2116
- Epoch 11 --- Training Accuracy: 71.63% --- Testing Accuracy: 70.26% --- Training Loss: 0.2039 --- Testing Loss: 0.2114
- Epoch 12 --- Training Accuracy: 71.68% --- Testing Accuracy: 70.26% --- Training Loss: 0.2035 --- Testing Loss: 0.2112
- Epoch 13 --- Training Accuracy: 71.76% --- Testing Accuracy: 70.36% --- Training Loss: 0.2031 --- Testing Loss: 0.2108
- Epoch 14 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2026 --- Testing Loss: 0.2104
- Epoch 15 --- Training Accuracy: 71.81% --- Testing Accuracy: 70.36% --- Training Loss: 0.2019 --- Testing Loss: 0.2099
- Epoch 16 --- Training Accuracy: 71.85% --- Testing Accuracy: 70.36% --- Training Loss: 0.2012 --- Testing Loss: 0.2094
- Epoch 17 --- Training Accuracy: 71.93% --- Testing Accuracy: 70.46% --- Training Loss: 0.2003 --- Testing Loss: 0.2088
- Epoch 18 --- Training Accuracy: 73.26% --- Testing Accuracy: 70.96% --- Training Loss: 0.1997 --- Testing Loss: 0.2084
- Epoch 19 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1989 --- Testing Loss: 0.2078
- Epoch 20 --- Training Accuracy: 73.31% --- Testing Accuracy: 71.06% --- Training Loss: 0.1981 --- Testing Loss: 0.2072
- Epoch 21 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1972 --- Testing Loss: 0.2066
- Epoch 22 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1963 --- Testing Loss: 0.2060
- Epoch 23 --- Training Accuracy: 73.43% --- Testing Accuracy: 71.16% --- Training Loss: 0.1952 --- Testing Loss: 0.2052
- Epoch 24 --- Training Accuracy: 73.48% --- Testing Accuracy: 71.16% --- Training Loss: 0.1943 --- Testing Loss: 0.2046
- Epoch 25 --- Training Accuracy: 78.92% --- Testing Accuracy: 75.30% --- Training Loss: 0.1933 --- Testing Loss: 0.2039
- Epoch 26 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1923 --- Testing Loss: 0.2033
- Epoch 27 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1914 --- Testing Loss: 0.2027
- Epoch 28 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1906 --- Testing Loss: 0.2022
- Epoch 29 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1899 --- Testing Loss: 0.2017
- Epoch 30 --- Training Accuracy: 79.01% --- Testing Accuracy: 75.40% --- Training Loss: 0.1893 --- Testing Loss: 0.2013

We can see that convergence is, sadly, slower than before, with algorithm reaching maximum accuracy score for both training and testing data on a higher epoch than the model that doesn't use parameter shifting. Loss scores also decrease in a lower rate than before. However, time execution is significantly better, as training process for 30 epochs took approximately the same time that initial model took for 20 epochs.


## 5. Solving Another Binary Classification Problem
Our VQC is, theoretically, able to solve binary classification problems that receive 3-bit bitstrings as their input, with minor modifications. Therefore, we select the **majority problem** to use our quantum circuit on. The majority problem, in that case, relies on the tailor-made majority condition that at least 2 bits of input bistrings are 1. So, the predicted labels should be 1, if at least two out of the three input bits are 1, and 0 otherwise. We will conventionally test our classifier, for this classification problem, using the same model we used in previous subtopic (which includes parameter shifting), as such:

**For 30 epochs and batch size of 10:** 

- Epoch 1 --- Training Accuracy: 85.06% --- Testing Accuracy: 86.92% --- Training Loss: 0.1610 --- Testing Loss: 0.1583
- Epoch 2 --- Training Accuracy: 85.48% --- Testing Accuracy: 87.48% --- Training Loss: 0.1577 --- Testing Loss: 0.1551
- Epoch 3 --- Training Accuracy: 85.91% --- Testing Accuracy: 87.67% --- Training Loss: 0.1540 --- Testing Loss: 0.1516
- Epoch 4 --- Training Accuracy: 86.19% --- Testing Accuracy: 87.85% --- Training Loss: 0.1501 --- Testing Loss: 0.1479
- Epoch 5 --- Training Accuracy: 86.41% --- Testing Accuracy: 87.85% --- Training Loss: 0.1461 --- Testing Loss: 0.1441
- Epoch 6 --- Training Accuracy: 88.61% --- Testing Accuracy: 88.98% --- Training Loss: 0.1418 --- Testing Loss: 0.1400
- Epoch 7 --- Training Accuracy: 88.76% --- Testing Accuracy: 89.16% --- Training Loss: 0.1373 --- Testing Loss: 0.1357
- Epoch 8 --- Training Accuracy: 88.97% --- Testing Accuracy: 89.35% --- Training Loss: 0.1328 --- Testing Loss: 0.1314
- Epoch 9 --- Training Accuracy: 98.01% --- Testing Accuracy: 97.22% --- Training Loss: 0.1281 --- Testing Loss: 0.1269
- Epoch 10 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1235 --- Testing Loss: 0.1226
- Epoch 11 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1189 --- Testing Loss: 0.1182
- Epoch 12 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1143 --- Testing Loss: 0.1138
- Epoch 13 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1098 --- Testing Loss: 0.1096
- Epoch 14 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1057 --- Testing Loss: 0.1057
- Epoch 15 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1017 --- Testing Loss: 0.1020
- Epoch 16 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0979 --- Testing Loss: 0.0984
- Epoch 17 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0940 --- Testing Loss: 0.0948
- Epoch 18 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0904 --- Testing Loss: 0.0913
- Epoch 19 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0868 --- Testing Loss: 0.0880
- Epoch 20 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0834 --- Testing Loss: 0.0848
- Epoch 21 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0802 --- Testing Loss: 0.0818
- Epoch 22 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0771 --- Testing Loss: 0.0790
- Epoch 23 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0741 --- Testing Loss: 0.0762
- Epoch 24 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0712 --- Testing Loss: 0.0736
- Epoch 25 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0684 --- Testing Loss: 0.0710
- Epoch 26 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0659 --- Testing Loss: 0.0687
- Epoch 27 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0635 --- Testing Loss: 0.0665
- Epoch 28 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0611 --- Testing Loss: 0.0644
- Epoch 29 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0589 --- Testing Loss: 0.0624
- Epoch 30 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0569 --- Testing Loss: 0.0606

The exact same model used for the parity problem leads to much better accuracy and loss function scores, already from the first epoch. However, it becomes evident that model has reached its max accuracy value alreadu from the 10th epoch (training data) and 9th (testing data). Let's run the same model but with different batch sizes.

**For 30 epochs and batch size of 5:**

- Epoch 1 --- Training Accuracy: 85.48% --- Testing Accuracy: 87.48% --- Training Loss: 0.1592 --- Testing Loss: 0.1565
- Epoch 2 --- Training Accuracy: 85.91% --- Testing Accuracy: 87.67% --- Training Loss: 0.1548 --- Testing Loss: 0.1524
- Epoch 3 --- Training Accuracy: 86.19% --- Testing Accuracy: 87.85% --- Training Loss: 0.1503 --- Testing Loss: 0.1480
- Epoch 4 --- Training Accuracy: 86.41% --- Testing Accuracy: 88.04% --- Training Loss: 0.1455 --- Testing Loss: 0.1435
- Epoch 5 --- Training Accuracy: 88.61% --- Testing Accuracy: 89.16% --- Training Loss: 0.1405 --- Testing Loss: 0.1388
- Epoch 6 --- Training Accuracy: 88.90% --- Testing Accuracy: 89.35% --- Training Loss: 0.1353 --- Testing Loss: 0.1338
- Epoch 7 --- Training Accuracy: 97.94% --- Testing Accuracy: 97.22% --- Training Loss: 0.1300 --- Testing Loss: 0.1288
- Epoch 8 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1247 --- Testing Loss: 0.1237
- Epoch 9 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1194 --- Testing Loss: 0.1187
- Epoch 10 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1144 --- Testing Loss: 0.1140
- Epoch 11 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1098 --- Testing Loss: 0.1096
- Epoch 12 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1051 --- Testing Loss: 0.1052
- Epoch 13 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1005 --- Testing Loss: 0.1008
- Epoch 14 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0961 --- Testing Loss: 0.0967
- Epoch 15 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0918 --- Testing Loss: 0.0927
- Epoch 16 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0878 --- Testing Loss: 0.0890
- Epoch 17 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0842 --- Testing Loss: 0.0856
- Epoch 18 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0809 --- Testing Loss: 0.0825
- Epoch 19 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0776 --- Testing Loss: 0.0795
- Epoch 20 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0745 --- Testing Loss: 0.0766
- Epoch 21 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0714 --- Testing Loss: 0.0737
- Epoch 22 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0684 --- Testing Loss: 0.0711
- Epoch 23 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0656 --- Testing Loss: 0.0685
- Epoch 24 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0631 --- Testing Loss: 0.0662
- Epoch 25 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0608 --- Testing Loss: 0.0641
- Epoch 26 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0584 --- Testing Loss: 0.0620
- Epoch 27 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0564 --- Testing Loss: 0.0601
- Epoch 28 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0544 --- Testing Loss: 0.0584
- Epoch 29 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0526 --- Testing Loss: 0.0567
- Epoch 30 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0510 --- Testing Loss: 0.0553

Here, we converge towards an optimal solution faster than before, with loss function scores having slightly lower scores.

**For 30 epochs and batch size of 50:**
 
- Epoch 1 --- Training Accuracy: 85.48% --- Testing Accuracy: 87.48% --- Training Loss: 0.1594 --- Testing Loss: 0.1567
- Epoch 2 --- Training Accuracy: 85.91% --- Testing Accuracy: 87.67% --- Training Loss: 0.1547 --- Testing Loss: 0.1522
- Epoch 3 --- Training Accuracy: 86.19% --- Testing Accuracy: 87.85% --- Training Loss: 0.1497 --- Testing Loss: 0.1475
- Epoch 4 --- Training Accuracy: 86.41% --- Testing Accuracy: 88.04% --- Training Loss: 0.1447 --- Testing Loss: 0.1427
- Epoch 5 --- Training Accuracy: 88.69% --- Testing Accuracy: 89.16% --- Training Loss: 0.1396 --- Testing Loss: 0.1379
- Epoch 6 --- Training Accuracy: 88.90% --- Testing Accuracy: 89.35% --- Training Loss: 0.1344 --- Testing Loss: 0.1329
- Epoch 7 --- Training Accuracy: 97.94% --- Testing Accuracy: 97.22% --- Training Loss: 0.1294 --- Testing Loss: 0.1282
- Epoch 8 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1245 --- Testing Loss: 0.1235
- Epoch 9 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1196 --- Testing Loss: 0.1189
- Epoch 10 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1148 --- Testing Loss: 0.1143
- Epoch 11 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1101 --- Testing Loss: 0.1099
- Epoch 12 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1054 --- Testing Loss: 0.1054
- Epoch 13 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.1009 --- Testing Loss: 0.1012
- Epoch 14 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0966 --- Testing Loss: 0.0972
- Epoch 15 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0925 --- Testing Loss: 0.0934
- Epoch 16 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0886 --- Testing Loss: 0.0897
- Epoch 17 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0849 --- Testing Loss: 0.0863
- Epoch 18 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0814 --- Testing Loss: 0.0830
- Epoch 19 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0779 --- Testing Loss: 0.0798
- Epoch 20 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0747 --- Testing Loss: 0.0768
- Epoch 21 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0717 --- Testing Loss: 0.0740
- Epoch 22 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0688 --- Testing Loss: 0.0714
- Epoch 23 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0661 --- Testing Loss: 0.0689
- Epoch 24 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0635 --- Testing Loss: 0.0665
- Epoch 25 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0610 --- Testing Loss: 0.0643
- Epoch 26 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0587 --- Testing Loss: 0.0622
- Epoch 27 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0566 --- Testing Loss: 0.0603
- Epoch 28 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0546 --- Testing Loss: 0.0585
- Epoch 29 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0527 --- Testing Loss: 0.0569
- Epoch 30 --- Training Accuracy: 98.15% --- Testing Accuracy: 97.22% --- Training Loss: 0.0510 --- Testing Loss: 0.0553

This execution revealed little to no differences in loss function scores, with accuracy scores remaining the same. 

# Task 3
The purpose of this task is to extend our classifier so that it works for our **5-dimensional** input point dataset. Every technical detail behind this implementation will be revealed in respective code cells. We will train our model with these **hyperparameters**:

- **Number of qubits (n_qubits)** = 4
- **Number of layers (num_layers)** = 3
- **Step size** = 0.001
- **Batch size** = 16 
- **Bias term** = 0
- **Epochs** = 30 
- **Optimizer** = Nesterov Momentum Optimizer
- **Loss function** = Binary Cross Entropy or Mean Squarred Error (MSE)

It's worth mentioning, here, that hyperparameter **batch_size** was selected based on a number of model executions. Its value above is the one that led to the best model training results for the 5 input parity problem and, thus, we kept it fixed to 16.
We will run our model *twice*, once for each loss function mentioned above, so that we get to see how our model reacts to this new data.

**Binary Cross Entropy**: 

- Epoch 1 --- Training Accuracy: 61.28% --- Testing Accuracy: 68.85% --- Training Loss: 0.7717 --- Testing Loss: 0.7969
- Epoch 2 --- Training Accuracy: 61.28% --- Testing Accuracy: 68.85% --- Training Loss: 0.7693 --- Testing Loss: 0.7938
- Epoch 3 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7671 --- Testing Loss: 0.7909
- Epoch 4 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7650 --- Testing Loss: 0.7882
- Epoch 5 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7631 --- Testing Loss: 0.7857
- Epoch 6 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7613 --- Testing Loss: 0.7834
- Epoch 7 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7595 --- Testing Loss: 0.7811
- Epoch 8 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7577 --- Testing Loss: 0.7787
- Epoch 9 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7559 --- Testing Loss: 0.7764
- Epoch 10 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7543 --- Testing Loss: 0.7742
- Epoch 11 --- Training Accuracy: 61.47% --- Testing Accuracy: 68.85% --- Training Loss: 0.7526 --- Testing Loss: 0.7720
- Epoch 12 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7508 --- Testing Loss: 0.7696
- Epoch 13 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7491 --- Testing Loss: 0.7674
- Epoch 14 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7473 --- Testing Loss: 0.7650
- Epoch 15 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7456 --- Testing Loss: 0.7627
- Epoch 16 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7440 --- Testing Loss: 0.7604
- Epoch 17 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7422 --- Testing Loss: 0.7581
- Epoch 18 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7405 --- Testing Loss: 0.7558
- Epoch 19 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7389 --- Testing Loss: 0.7536
- Epoch 20 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7374 --- Testing Loss: 0.7515
- Epoch 21 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7359 --- Testing Loss: 0.7495
- Epoch 22 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7344 --- Testing Loss: 0.7475
- Epoch 23 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7329 --- Testing Loss: 0.7454
- Epoch 24 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7314 --- Testing Loss: 0.7432
- Epoch 25 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7298 --- Testing Loss: 0.7411
- Epoch 26 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7284 --- Testing Loss: 0.7393
- Epoch 27 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7271 --- Testing Loss: 0.7376
- Epoch 28 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7259 --- Testing Loss: 0.7360
- Epoch 29 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7247 --- Testing Loss: 0.7344
- Epoch 30 --- Training Accuracy: 61.67% --- Testing Accuracy: 68.85% --- Training Loss: 0.7235 --- Testing Loss: 0.7329

We can see that testing accuracy stays the same for each epoch, whereas training accuracy scores exhibits relatively marginal changes in values. Both training and testing loss, on the other hand, decrease, as the number of epochs increases. 

**Mean Squarred Error (MSE)**:

- Epoch 1 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4093 --- Testing Loss: 0.3298
- Epoch 2 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4108 --- Testing Loss: 0.3307
- Epoch 3 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4118 --- Testing Loss: 0.3313
- Epoch 4 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4123 --- Testing Loss: 0.3316
- Epoch 5 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4126 --- Testing Loss: 0.3318
- Epoch 6 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4123 --- Testing Loss: 0.3316
- Epoch 7 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4121 --- Testing Loss: 0.3314
- Epoch 8 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4110 --- Testing Loss: 0.3307
- Epoch 9 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4096 --- Testing Loss: 0.3297
- Epoch 10 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4077 --- Testing Loss: 0.3285
- Epoch 11 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4055 --- Testing Loss: 0.3270
- Epoch 12 --- Training Accuracy: 57.26% --- Testing Accuracy: 62.16% --- Training Loss: 0.4032 --- Testing Loss: 0.3255
- Epoch 13 --- Training Accuracy: 57.26% --- Testing Accuracy: 63.29% --- Training Loss: 0.4006 --- Testing Loss: 0.3238
- Epoch 14 --- Training Accuracy: 57.78% --- Testing Accuracy: 63.29% --- Training Loss: 0.3980 --- Testing Loss: 0.3221
- Epoch 15 --- Training Accuracy: 57.58% --- Testing Accuracy: 63.29% --- Training Loss: 0.3955 --- Testing Loss: 0.3205
- Epoch 16 --- Training Accuracy: 57.39% --- Testing Accuracy: 63.29% --- Training Loss: 0.3933 --- Testing Loss: 0.3191
- Epoch 17 --- Training Accuracy: 57.39% --- Testing Accuracy: 63.29% --- Training Loss: 0.3913 --- Testing Loss: 0.3179
- Epoch 18 --- Training Accuracy: 57.39% --- Testing Accuracy: 62.67% --- Training Loss: 0.3893 --- Testing Loss: 0.3166
- Epoch 19 --- Training Accuracy: 57.39% --- Testing Accuracy: 62.67% --- Training Loss: 0.3871 --- Testing Loss: 0.3152
- Epoch 20 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3850 --- Testing Loss: 0.3139
- Epoch 21 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3830 --- Testing Loss: 0.3127
- Epoch 22 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3811 --- Testing Loss: 0.3115
- Epoch 23 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3789 --- Testing Loss: 0.3102
- Epoch 24 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3768 --- Testing Loss: 0.3089
- Epoch 25 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3748 --- Testing Loss: 0.3077
- Epoch 26 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3726 --- Testing Loss: 0.3064
- Epoch 27 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3706 --- Testing Loss: 0.3052
- Epoch 28 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3687 --- Testing Loss: 0.3041
- Epoch 29 --- Training Accuracy: 57.19% --- Testing Accuracy: 62.67% --- Training Loss: 0.3669 --- Testing Loss: 0.3030
- Epoch 30 --- Training Accuracy: 56.81% --- Testing Accuracy: 62.67% --- Training Loss: 0.3649 --- Testing Loss: 0.3018

The usage of an alternative loss function led to fluctuations in both training and testing accuracy values. However, we can spot noticeable differences in loss scores in both training and testing data, which was expected since this loss function gave the best loss scores in previous comparisons (using the nesterov momentum optimizer). 
