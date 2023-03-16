import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)

    mse_values = {}
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_eval)

        mse = np.mean((y_pred - y_eval) ** 2)
        mse_values[tau] = mse
        print(f'tau={tau}, MSE={mse}')

        plt.figure()
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_eval, y_pred, 'go', linewidth=2)
        plt.title(f'MSE (with tau = {tau}): {mse}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/p05b_{tau}.png')

    best_tau = min(mse_values, key=mse_values.get)
    print(mse_values)

    model = LocallyWeightedLinearRegression(tau=best_tau)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = np.mean((y_pred - y_test) ** 2)
    print(f'Best with tau={best_tau}, resulting MSE on test set: {mse}')

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_pred, 'go', linewidth=2)
    plt.title(f'MSE (with tau = {best_tau}): {mse}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'output/p05c_best_{best_tau}.png')

    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***
