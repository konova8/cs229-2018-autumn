import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n+1)

        y1 = sum(y)
        y0 = m - y1

        phi = y1 / m
        mu0 = sum([x[i] for i in range(0, m) if y[i] == 0]) / y0
        mu1 = sum([x[i] for i in range(0, m) if y[i] == 1]) / y1
        #sigma = (1 / m) * sum([(x[i] - (mu0 ** (1 - y[i]) * mu1 ** y[i])).dot((x[i] - (mu0 ** (1 - y[i]) * mu1 ** y[i])).T) for i in range(1, m)]) / m
        sigma = ((x[y == 0] - mu0).T.dot(x[y == 0] - mu0) + (x[y == 1] - mu1).T.dot(x[y == 1] - mu1)) / m

        sigma_inv = np.linalg.inv(sigma)

        self.theta[0] = 0.5 * (mu0 + mu1).dot(sigma_inv).dot(mu0 - mu1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu1 - mu0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE
