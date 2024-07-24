import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.m = 0
        self.b = 0
        self.history = {'m': [], 'b': [], 'error': []}
        self.paused = False

    def fit(self, X, y, visualize=False):
        if visualize:
            plt.ion()  # Turn on interactive mode
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.scatter(X, y, color='blue')
            line, = ax1.plot(X, self.predict(X), color='red')
            error_line, = ax2.plot([], [], color='green')
            plt.xlabel('X')
            plt.ylabel('y')
            ax1.set_xlim(min(X), max(X))
            ax1.set_ylim(min(y), max(y))
            ax2.set_xlim(0, self.num_iterations)
            ax2.set_ylim(0, max(y)**2)
            ax1.set_title('Linear Regression Fit')
            ax2.set_title('Error Over Iterations')
            ax1.set_xlabel('X')
            ax1.set_ylabel('y')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Error')

            # Create a color map for the iterations
            cmap = plt.get_cmap('viridis')
            color_idx = np.linspace(0, 1, self.num_iterations)

            # Add pause/resume buttons
            pause_ax = plt.axes([0.81, 0.01, 0.1, 0.05])
            resume_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
            pause_button = plt.Button(pause_ax, 'Pause')
            resume_button = plt.Button(resume_ax, 'Resume')
            pause_button.on_clicked(self.pause)
            resume_button.on_clicked(self.resume)

        for i in range(self.num_iterations):
            while self.paused:
                plt.pause(0.1)
            self.gradient_descent_step(X, y)
            y_pred = self.predict(X)
            error = self.mse(y, y_pred)
            self.history['m'].append(self.m)
            self.history['b'].append(self.b)
            self.history['error'].append(error)
            if visualize and i % 10 == 0:  # Update plot every 10 iterations
                line.set_ydata(y_pred)
                line.set_color(cmap(color_idx[i]))
                error_line.set_data(range(i+1), self.history['error'])
                ax2.set_ylim(0, max(self.history['error']))
                ax1.set_title(f"Iteration {i}, Error: {error:.4f}, m: {self.m:.4f}, b: {self.b:.4f}")
                plt.pause(0.1)

            if i % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {i}, Error: {error:.4f}, m: {self.m:.4f}, b: {self.b:.4f}")

        if visualize:
            plt.ioff()  # Turn off interactive mode
            plt.show()

        return self

    def predict(self, X):
        return [self.m * x + self.b for x in X]

    def mse(self, y_true, y_pred):
        return sum((pred - actual) ** 2 for pred, actual in zip(y_pred, y_true)) / len(y_true)

    def gradient_descent_step(self, X, y):
        n = len(X)
        y_pred = self.predict(X)

        dm = (1/n) * sum((pred - actual) * x for x, pred, actual in zip(X, y_pred, y))
        db = (1/n) * sum(pred - actual for pred, actual in zip(y_pred, y))

        self.m -= self.learning_rate * dm
        self.b -= self.learning_rate * db

    def pause(self, event):
        self.paused = True

    def resume(self, event):
        self.paused = False

class LinearRegressionNumpy:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.m = None
        self.b = None

    def fit(self, X, y):
        # Ensure X is a 2D array
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        # Initialize parameters
        self.m = np.zeros((X.shape[1], 1))
        self.b = 0

        for _ in range(self.num_iterations):
            self.gradient_descent_step(X, y)

        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return np.dot(X, self.m) + self.b

    def mse(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def gradient_descent_step(self, X, y):
        n = X.shape[0]
        y_pred = self.predict(X)

        dw = (1/n) * np.dot(X.T, (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)

        self.m -= self.learning_rate * dw
        self.b -= self.learning_rate * db

def time_model(model, X, y, name):
    start = time.time()
    model.fit(X, y)
    end = time.time()
    print(f"{name} took {end - start:.4f} seconds")

def load_data(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        header = file.readline()  # Skip the header line
        for line in file:
            values = line.strip().split(',')
            x.append(float(values[0]))
            y.append(float(values[1]))
    return x, y

def split_data(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    return list(X_train), list(y_train), list(X_test), list(y_test)

def main():
    # Load data
    X, y = load_data('./tvmarketing.csv')

    # Split data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(X, y, test_size=0.2)

    # Check for visualization argument
    visualize = 'visualize' in sys.argv

    # Create and train the NumPy model
    model_np = LinearRegressionNumpy(learning_rate=0.0001, num_iterations=1000)
    model_np.fit(X_train, y_train)

    # Create and train the Python model
    model_py = LinearRegression(learning_rate=0.0001, num_iterations=1000)
    model_py.fit(X_train, y_train, visualize=visualize)

    # Print final models
    print(f"Python model: y = {model_py.m}x + {model_py.b}")
    print(f"NumPy model: y = {model_np.m[0][0]}x + {model_np.b}")

    # Evaluate the models on the test set
    y_test_pred_py = model_py.predict(X_test)
    test_error_py = model_py.mse(y_test, y_test_pred_py)
    print(f"Test Error (Python): {test_error_py:.4f}")

    y_test_pred_np = model_np.predict(X_test)
    test_error_np = model_np.mse(np.array(y_test).reshape(-1, 1), y_test_pred_np)
    print(f"Test Error (NumPy): {test_error_np:.4f}")

    time_model(model_py, X, y, "Python model")
    time_model(model_np, X, y, "NumPy model")

if __name__ == "__main__":
    main()
