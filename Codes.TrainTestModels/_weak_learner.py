"""
Soft Voting/Majority Rule classifier.

This module contains a Soft Voting/Majority Rule classifier for
classification estimators.

"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted


class VotingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <voting_classifier>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array-like, shape = [n_predictions]
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """

    def __init__(self, estimators, partitions, weigh_num_feats=0, voting='hard', missing_features=[]):

        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.voting = voting        
        self.partitions = partitions
        self.weights = self._calculate_partition_weight(partitions, weigh_num_feats)
        self.missing_features = missing_features
        


    # Samaneh
    def _partition_data(self, X, partition):
        """ Partition data X based on the provided partitioning """
        X_list = []  
        missing_set = set(self.missing_features)
        for part in partition:
            if len(set(part).intersection(missing_set)) > 0:
                X_list.append([])
            else:
                X_list.append(X[part])
        return X_list

    # Samaneh
    def _calculate_partition_weight(self, partitions, weigh_num_features):
        """ Weight of each partition is eqaul to number of features in that partition. """
        if weigh_num_features == 1:
            weights = []
            for part in partitions:
                weights.append(len(part))
            return weights
        else:
            return [1] * len(partitions)


    def fit(self, X, y):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """        
                
        
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []
        
        X_list = self._partition_data(X, self.partitions)        
        
#         if (self.weights is not None):
#             new_weights = []
#             for i in range(0, len(self.weights)):
#                 X = X_list[i]
#                 if len(X) > 0:
#                     new_weights.append(self.weights[i])
#                     
#             self.weights = new_weights
#         
        
#         for name, clf in self.estimators:
        for i in range(0, len(self.estimators)):
            name = self.estimators[i][0]
            clf = self.estimators[i][1]
            X = X_list[i]
            if len(X) == 0:
                continue

            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.estimators_.append(fitted_clf)
            
        return self
    

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        
        X_list = self._partition_data(X, self.partitions) 
        
        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X_list), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X_list)
            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions.astype('int'))

        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X_list):
        """Collect results from clf.predict calls. """     
        probs = []
        for i in range(0, len(self.estimators_)):
            clf = self.estimators_[i]
            X = X_list[i]
            p = clf.predict_proba(X)
            probs.append(p)
        return np.asarray(probs)

    def _predict_proba(self, X_list):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X_list), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        X_list = self._partition_data(X, self.partitions)
        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            return self._collect_probas(X_list)
        else:
            return self._predict(X_list)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(VotingClassifier, self).get_params(deep=False)
        else:
            out = super(VotingClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X_list):
        """Collect results from clf.predict calls. """
        results = []
        for i in range(0, len(self.estimators_)):
            clf = self.estimators_[i]
            X = X_list[i]
            r = clf.predict(X)
            results.append(r)
        return np.asarray(results).T
    
