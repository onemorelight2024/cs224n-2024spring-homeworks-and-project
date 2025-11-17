# cs224n-2024spring-homeworks-and-project
For final project：
The base model suffered from severe overfitting, so I applied Smoothness-Inducing Adversarial Regularization (SIAR) to improve generalization.
In addition, to support multi-task learning, I implemented Gradient Surgery and introduced task-specific projection heads for each task.

Under the hyperparameter setting lr = 1e-5, epoch = 15, eps = 0.15, coeff = 1.0, proj = 384, the model achieved:

dev sentiment acc: 0.540
dev paraphrase acc: 0.600
dev STS corr: 0.670

However, due to the paraphrase dataset being significantly larger than the other two, and because Gradient Surgery requires all three tasks to update synchronously, I did not load the paraphrase training set.
The model was trained only on the other two tasks’ datasets, which may represent a key direction for further improvement.
