def mse(y_true, y_pred):
    n = len(y_true)
    sum_of_squared_errors = 0
    for actual, predicted in zip(y_true, y_pred):
        error = predicted - actual
        squared_error = error ** 2
        sum_of_squared_errors += squared_error
    return sum_of_squared_errors / n

def gradient_descent_step(x, y, m, b, learning_rate):
    n = len(x)

    # Calculate predictions
    y_pred = [m * xi + b for xi in x]

    # Calculate gradients
    dm = (1/n) * sum((pred - actual) * xi for pred, actual, xi in zip(y_pred, y, x))
    db = (1/n) * sum(pred - actual for pred, actual in zip(y_pred, y))

    # Update m and b
    new_m = m - learning_rate * dm
    new_b = b - learning_rate * db

    return new_m, new_b

def predict(x, m, b):
    """
    Predict y values for given x values using the line equation y = mx + b

    :param x: A single x value or a list of x values
    :param m: The slope of the line
    :param b: The y-intercept of the line
    :return: The predicted y value(s)
    """
    if isinstance(x, (list, tuple)):
        return [m * xi + b for xi in x]
    else:
        return m * x + b

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

def main():
    # Load data
    x, y = load_data('./tvmarketing.csv')  # Assume we have a file named 'data.csv'

    # Set hyperparameters
    learning_rate = 0.000001
    num_iterations = 10000

    # Initialize model parameters
    m = 0
    b = 0

    # Train the model
    for i in range(num_iterations):
        y_pred = predict(x, m, b)
        error = mse(y, y_pred)
        m, b = gradient_descent_step(x, y, m, b, learning_rate)

        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}, Error: {error}, m: {m}, b: {b}")

    # Print final model
    print(f"Final model: y = {m}x + {b}")

    # Make a prediction
    x_test = 5  # Example test value
    y_pred = predict(x_test, m, b)
    print(f"Prediction for x = {x_test}: {y_pred}")

if __name__ == "__main__":
    main()
