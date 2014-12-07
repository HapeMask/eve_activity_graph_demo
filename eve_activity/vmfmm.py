import numpy as np
from scipy.special import iv
from sklearn.utils import check_random_state
from sklearn.utils.extmath import logsumexp
from sklearn.base import  BaseEstimator

######################################
# Von-Mises Fisher Utility Functions
######################################
def log_norm_constant(precs, p):
    """Compute the log normalization constant for a set of von-Mises Fisher
    (vMF) distributions.

    Parameters
    ----------
    precs : array_like
        List of n_components concentration parameters for each vMF distribution.

    Returns
    -------
    Cpk : array_like, shape (n_components,)
        Array containing the log normalization constants of each of the
        n_components vMF distributions.
    """
    if p == 3:
        return (np.log(precs) - np.log(2*np.pi) -
                np.log(np.exp(precs) - np.exp(-precs)))
    else:
        return ((p/2 - 1) * np.log(precs) -
                ((p/2)*np.log((2*np.pi)) + np.log(iv((p/2)-1, precs))))

def log_vmf_pdf(X, means, precs):
    """Compute the log probability under a set of von-Mises Fisher (vMF)
    distributions.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components vMF distributions.
        Each row corresponds to a single mean vector.

    precs : array_like
        List of n_components concentration parameters for each vMF distribution.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components vMF distributions.
    """
    return (log_norm_constant(precs, means.shape[1]) +
            precs*np.dot(X, means.T))

def check_angular(X):
    """Check an array to see if it is 1-dimensional. If so, transform the list
    of angles to their 2D cartesian coordinates.

    Parameters
    ----------
    X : array_like, shape (n_samples,)
        List of angles (in radians).

    Returns
    -------
    rect : array_like, shape (n_samples, 2)
        Array containing the 2D rectangular coordinates for the given angles.
    """

    if X.ndim == 1:
        X = X[:, np.newaxis]

    if X.shape[1] == 1:
        X = np.hstack([np.cos(X), np.sin(X)])

    return X

def sample_vmf_canonical(concentration, size=(1,2), random_state=None):
    """Sample from the canonical von-Mises Fisher (vMF) distribution.
    This is the vMF distribution with the given concentration, and with
    mean = (0, 0, ..., 1).

    Parameters
    ----------
    concentration : scalar
        Concentration parameter for the vMF distribution.

    size : tuple, (n_samples, n_features)
        Number of samples and dimensionality of the sample space.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance.

    Returns
    -------
    samples : array_like, shape (n_samples, n_features)
        Samples from the canonical vMF distribution.
    """

    random_state = check_random_state(random_state)

    # If concentration is ~0, the distribution is the uniform distribution on
    # the unit sphere.
    if concentration < 1e-8:
        samples = random_state.normal(0, 1, size=size)
        samples /= np.linalg.norm(samples, axis=-1)[:, np.newaxis]
        return samples

    # Sample the unit dim-1-sphere uniformly.
    samples = random_state.normal(0, 1, size=(size[0], size[1]-1))
    samples /= np.linalg.norm(samples, axis=-1)[:, np.newaxis]

    # Sample scale factors.
    uniforms = random_state.uniform(size=size[0])
    scale = 1 + (1/concentration) * (
            np.log(uniforms) +
            np.log(1 - ((uniforms-1)/uniforms)*np.exp(-2*concentration))
            )[:, np.newaxis]

    return np.hstack([np.sqrt(1 - scale**2)*samples, scale])

def sample_vmf(mean, concentration, size=1, random_state=None):
    """Sample from the von-Mises Fisher (vMF) distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        The mean vector for the vMF distribution. Must be a unit vector.

    concentration : scalar
        Concentration parameter for the vMF distribution.

    size : int, 
        Number of samples to generate.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance.

    Returns
    -------
    samples : array_like, shape (n_samples, n_features)
        Samples from the given vMF distribution.
    """
    mean = np.asarray(mean, dtype=np.float)
    random_state = check_random_state(random_state)
    n_features = mean.shape[0]

    # Sample the canonical vMF distribution w/the given concentration.
    samples = sample_vmf_canonical(concentration, (size, n_features), random_state)

    # Rotation is meaningless for a uniform spherical distribution.
    if concentration < 1e-8:
        return samples

    # Compute the rotation matrix from the canonical mean to the given mean.
    cos_a = mean[-1]
    sin_a = np.sin(np.arccos(cos_a))
    if n_features == 2:
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
    elif n_features == 3:
        u = np.cross((0,0,1), mean)
        u /= np.linalg.norm(u)
        tensor_mat = np.outer(u, u)
        cross_mat = np.array([[0,    -u[2], u[1]],
                              [u[2],  0,   -u[0]],
                              [-u[1], u[0], 0]])
        R = cos_a * np.eye(n_features) + sin_a * cross_mat + (1-cos_a) * tensor_mat
    else:
        raise NotImplementedError("Sampling not implemented for more than 3 dimensions.")

    return np.dot(samples, R.T)

class VMFMM(BaseEstimator):
    """von-Mises Fisher Mixture Model

    Representation of a von-Mises Fisher mixture model probability
    distribution. This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a vMFMM distribution.


    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of different initializations to try. Attributes from the best
        run are kept.

    thresh : float, optional
        Convergence threshold.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    precs_ : array
        Concentration parameters for each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    """

    def __init__(self, n_components, n_iter=100, n_init=1, thresh=1e-3, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        self.thresh = thresh
        self.random_state = random_state
        self.converged_ = False

    def fit(self, X):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to
            a single data point.
        """
        X = np.asarray(X, dtype=np.float)
        X = check_angular(X)

        N, self.n_features = X.shape

        max_log_prob = -np.inf
        for ini in range(self.n_init):
            # Initialize means w/uniform random samples from unit sphere.
            random_state = check_random_state(self.random_state)
            self.means_ = random_state.normal(0,1,size=(self.n_components, self.n_features))
            self.means_ /= np.linalg.norm(self.means_, axis=-1)[:, np.newaxis]

            self.weights_ = np.ones(self.n_components) / self.n_components
            self.precs_ = np.ones(self.n_components)

            logprobs = []
            for it in range(self.n_iter):
                # E step
                logprob, responsibilities = self.score_samples(X)
                logprobs.append(logprob.sum())

                if it > 0 and abs(logprobs[-1] - logprobs[-2]) < self.thresh:
                    self.converged_ = True
                    break

                # M step
                self.weights_[:] = responsibilities.mean(axis=0)
                self.means_[:] = (X[:,np.newaxis] * responsibilities[:,:,np.newaxis]).sum(axis=0)
                mean_norms = np.linalg.norm(self.means_, axis=-1)
                r = mean_norms / (N*self.weights_)
                self.means_ /= mean_norms[:, np.newaxis]
                self.precs_ = r*(self.n_features - r**2) / (1 - r**2)

            if logprobs[-1] > max_log_prob:
                max_log_prob = logprobs[-1]
                best_params = {"weights": self.weights_,
                               "means":   self.means_,
                               "precs":   self.precs_}
        return self

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = check_angular(X)

        logprobs = (log_vmf_pdf(X, self.means_, self.precs_) +
                    np.log(self.weights_[np.newaxis]))
        logprob = logsumexp(logprobs, axis=1)

        responsibilities = np.exp(logprobs - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        X = check_angular(X)

        return self.score_samples(X)[0]

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        C : array-like, shape (n_samples,)
            Labels for the data points indicating which component of the model
            most likely corresponds to each point.
        """
        X = check_angular(X)

        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each vMF distribution in
        the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array-like, shape (n_samples, n_components)
            Returns the probability of the sample for each vMF in the model.
        """
        X = check_angular(X)

        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        weight_cdf = np.cumsum(self.weights_)

        X = np.empty((n_samples, self.means_.shape[1]))
        rand = random_state.rand(n_samples)

        # Decide which component to use for each sample.
        comps = weight_cdf.searchsorted(rand)

        # For each component, generate all needed samples.
        for comp in range(self.n_components):
            # Occurrences of current component in X
            comp_in_X = (comp == comps)

            # Number of those occurrences
            num_comp_in_X = comp_in_X.sum()

            if num_comp_in_X > 0:
                X[comp_in_X] = sample_vmf(
                    self.means_[comp], self.precs_[comp],
                    size=num_comp_in_X, random_state=random_state)

        return X
