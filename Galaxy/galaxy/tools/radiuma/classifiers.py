from pandas.core.common import random_state
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
import logging
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


class Classifiers:

    def __init__(self, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
        # self.AlgName = AlgName
        # self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.regressors = {
            "BayesianRidge": BayesianRidge,
            "GaussianProcessRegressor": GaussianProcessRegressor,
            "KernelRidge": lambda: KernelRidge(kernel="rbf"),
            "KNeighborsRegressor": KNeighborsRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "LinearRegression": LinearRegression,
            "Lasso": Lasso,
            "Ridge": Ridge,
            "ElasticNet": ElasticNet,
            "SVR": SVR,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }

    def Vs_fit_and_predict_model(self):
        try:
            # Validate input dimensions
            if self.X_train is None or self.X_train.shape[0] == 0 or self.X_train.shape[1] == 0:
                print(f"Error: Invalid X_train dimensions: {self.X_train.shape if self.X_train is not None else 'None'}")
                return None, None, None

            # Convert y_train to 1D array if it's not None
            if self.y_train is not None:
                if hasattr(self.y_train, 'values'):
                    y_train_1d = self.y_train.values.ravel()
                elif hasattr(self.y_train, 'ravel'):
                    y_train_1d = self.y_train.ravel()
                else:
                    y_train_1d = np.array(self.y_train).ravel()

                # Validate dimensions match
                if len(y_train_1d) != self.X_train.shape[0]:
                    print(f"Error: Dimension mismatch - X_train: {self.X_train.shape[0]} samples, y_train: {len(y_train_1d)} labels")
                    return None, None, None

                self.model.fit(self.X_train, y_train_1d)
            else:
                # Handle case when y_train is None
                return None, None, None

            # Validate X_val dimensions before prediction
            if self.X_val is not None and self.X_val.shape[1] != self.X_train.shape[1]:
                print(f"Error: Feature count mismatch - X_train: {self.X_train.shape[1]}, X_val: {self.X_val.shape[1]}")
                return None, None, None

            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val) if self.X_val is not None else None

            if self.X_test is not None:
                # Validate X_test dimensions before prediction
                if self.X_test.shape[1] != self.X_train.shape[1]:
                    print(f"Error: Feature count mismatch - X_train: {self.X_train.shape[1]}, X_test: {self.X_test.shape[1]}")
                    y_test_pred = None
                else:
                    y_test_pred = self.model.predict(self.X_test)
            else:
                y_test_pred = None

            return y_train_pred, y_val_pred, y_test_pred

        except Exception as e:
            print(f"Error in model fitting/prediction: {e}")
            return None, None, None

    def _safe_extract_cm_elements(self, cm):
        """
        Safely extract tn, fp, fn, tp values from confusion matrix of any size.
        Handles 1x1, 2x2, and larger confusion matrices appropriately.

        Args:
            cm: confusion matrix from sklearn.metrics.confusion_matrix

        Returns:
            tuple: (tn, fp, fn, tp) with appropriate defaults for missing elements
        """
        try:
            # Get matrix dimensions
            rows, cols = cm.shape

            # Log unusual matrix dimensions for debugging
            if rows == 1 and cols == 1:
                logging.warning(f"Single-class prediction detected. Confusion matrix shape: {cm.shape}, matrix: {cm}")
                # All predictions are the same class
                # tn = count of correct predictions, others are 0
                return cm[0, 0], 0, 0, 0
            elif rows != 2 or cols != 2:
                logging.info(f"Non-standard confusion matrix dimensions: {cm.shape}")

            # For 2x2 matrix (standard binary classification)
            if rows >= 2 and cols >= 2:
                # Extract standard binary classification metrics
                tn = cm[0, 0] if rows > 0 and cols > 0 else 0
                fp = cm[0, 1] if rows > 0 and cols > 1 else 0
                fn = cm[1, 0] if rows > 1 and cols > 0 else 0
                tp = cm[1, 1] if rows > 1 and cols > 1 else 0
                return tn, fp, fn, tp

            # For other edge cases, return zeros
            else:
                logging.warning(f"Unusual confusion matrix shape {cm.shape}, returning zero values")
                return 0, 0, 0, 0

        except Exception as e:
            # Log the error and return safe defaults
            logging.error(f"Error extracting confusion matrix elements: {e}")
            return 0, 0, 0, 0

    def Vs_Score_cls(self, y, y_pred):
        """
        Calculate classification scores with robust confusion matrix handling.

        Args:
            y: true labels
            y_pred: predicted labels

        Returns:
            dict: classification metrics including tn, fp, fn, tp, accuracy, etc.
        """
        try:
            # Input validation
            if len(y) == 0 or len(y_pred) == 0:
                logging.warning("Empty input arrays for classification scoring")
                return {
                    'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
                    'acc': 0.0, 'MisclassificationRate': 1.0, 're': 0.0,
                    'Sensitivity': 0.0, 'pre': 0.0, 'f_sc': 0.0, 'AUC': 0.0, 'Specificity': 0.0
                }

            # Check for single-class scenarios
            unique_y = np.unique(y)
            unique_y_pred = np.unique(y_pred)

            if len(unique_y_pred) == 1:
                logging.warning(f"Single-class prediction detected: all predictions are class {unique_y_pred[0]}")

            cm = confusion_matrix(y, y_pred)

            ac = accuracy_score(y, y_pred)
            MisclassificationRate = 1 - ac

            # Safely extract confusion matrix elements
            tn, fp, fn, tp = self._safe_extract_cm_elements(cm)

            # Calculate metrics with zero division handling
            re = recall_score(y, y_pred, average='weighted', zero_division=0)
            Sensitivity = re

            pr = precision_score(y, y_pred, average='weighted', zero_division=0)
            fs = f1_score(y, y_pred, average='weighted', zero_division=0)

            # Calculate Specificity
            if tn + fp > 0:
                Specificity = tn / (tn + fp)
            else:
                Specificity = 0.0

            # Calculate AUC (for binary classification)
            try:
                from sklearn.metrics import roc_auc_score
                if len(unique_y) == 2 and len(unique_y_pred) == 2:
                    auc = roc_auc_score(y, y_pred)
                else:
                    auc = 0.0
            except Exception as e:
                logging.warning(f"Could not calculate AUC: {e}")
                auc = 0.0

            # Log successful completion for debugging
            logging.debug(f"Classification metrics calculated successfully. Accuracy: {ac:.3f}")

            return {
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'acc': ac, 'MisclassificationRate': MisclassificationRate, 're': re,
                'Sensitivity': Sensitivity, 'pre': pr, 'f_sc': fs, 'AUC': auc, 'Specificity': Specificity
            }

        except Exception as e:
            logging.error(f"Error in Vs_Score_cls: {e}", exc_info=True)
            # Return safe defaults to prevent crashes
            return {
                'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
                'acc': 0.0, 'MisclassificationRate': 1.0, 're': 0.0,
                'Sensitivity': 0.0, 'pre': 0.0, 'f_sc': 0.0, 'AUC': 0.0, 'Specificity': 0.0
            }

    def Vs_Score_cls_All_Data(self, y_train_pred, y_val_pred, y_test_pred):

        # Check if y_train is already a numpy array or has to_numpy method
        y_train_np = self.y_train if isinstance(self.y_train, np.ndarray) else (self.y_train.to_numpy() if hasattr(self.y_train, 'to_numpy') else np.array(self.y_train))
        train_score_dict = self.Vs_Score_cls(y_train_np, y_train_pred)

        # Check if y_val is already a numpy array or has to_numpy method
        y_val_np = self.y_val if isinstance(self.y_val, np.ndarray) else (self.y_val.to_numpy() if hasattr(self.y_val, 'to_numpy') else np.array(self.y_val))
        val_score_dict = self.Vs_Score_cls(y_val_np, y_val_pred)

        if y_test_pred is not None and self.y_test is not None:
            # Check if y_test is already a numpy array or has to_numpy method
            y_test_np = self.y_test if isinstance(self.y_test, np.ndarray) else (self.y_test.to_numpy() if hasattr(self.y_test, 'to_numpy') else np.array(self.y_test))
            test_score_dict = self.Vs_Score_cls(y_test_np, y_test_pred)
        else:
            test_score_dict = None

        return train_score_dict, val_score_dict, test_score_dict

    # def SelectAlg(self):
    #     import json
    #     f = open('data.json')
    #     JSONdata = json.load(f)
    #     f.close()
    #     algn = self.AlgName
    #     if self.AlgName == "LogisticRegression":

    #         self.Vs_LogisticClassifier(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1],
    #             JSONdata[algn][2],
    #             JSONdata[algn][3],
    #             JSONdata[algn][4],
    #             JSONdata[algn][5]
    #         )
    #     elif self.AlgName == "BaggingClassifier":
    #         self.Vs_BaggingClassifier(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1],
    #             JSONdata[algn][2],
    #             JSONdata[algn][3],
    #             JSONdata[algn][4]
    #         )
    #     elif self.AlgName == "AdaBoostClassifier":
    #         self.Vs_AdaBoostClassifier(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1],
    #             JSONdata[algn][2],
    #             JSONdata[algn][3],
    #             JSONdata[algn][4]
    #         )
    #     elif self.AlgName == "KNeighborsClassifier":
    #         self.Vs_KNeighborsClassifier(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1],
    #             JSONdata[algn][2],
    #             JSONdata[algn][3],
    #             JSONdata[algn][4],
    #             JSONdata[algn][5]
    #         )
    #     else:
    #         print("Algorithm name is wrong")

    def Vs_LogisticClassifier(self,
                              penalty="l2",
                              C=1.0,
                              class_weight=None,
                              solver="lbfgs",
                              max_iter=100,
                              multi_class="auto",
                              random_state=None
                              ):

        solver = solver.lower()
        multi_class = multi_class.lower()

        if penalty == "None" or penalty is None:
            penalty = None
        else:
            penalty = penalty.lower()

        if class_weight == "None" or class_weight is None:
            class_weight = None
        else:
            class_weight = class_weight.lower()

        model = LogisticRegression(
            penalty =penalty,
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state
        )
        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()

        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_BaggingClassifier(self,
                             estimator=None,
                             n_estimators=10,
                             max_samples=1.0,
                             max_features=1.0,
                             bootstrap=True,
                             random_state=None):

        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        if bootstrap == "True":
            bootstrap = True
        else:
            bootstrap = False

        model = BaggingClassifier(
            estimator=estimator_instance,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()

        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                                      y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_AdaBoostClassifier(self,
                              estimator=None,
                              n_estimators=50,
                              learning_rate=1.0,
                              algorithm="SAMME.R",
                              random_state=None):

        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        model = AdaBoostClassifier(
            estimator=estimator_instance,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )
        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()
        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_KNeighborsClassifier(self,
                                n_neighbors=5,
                                weights="uniform",
                                algorithm="auto",
                                p=2,
                                metric="minkowski"):

        if weights is not None:
            weights = weights.lower()

        algorithm = algorithm.lower()
        metric = metric.lower()

        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     weights=weights,
                                     algorithm=algorithm,
                                     p=p,
                                     metric=metric)
        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()

        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_LogisticClassifier_pipeline(self,
                                       penalty="l2",
                                       C=1.0,
                                       class_weight=None,
                                       solver="lbfgs",
                                       max_iter=100,
                                       multi_class="auto",
                                       random_state=None
                                       ):

        solver = solver.lower()
        multi_class = multi_class.lower()

        if penalty == "None" or penalty is None:
            penalty = None
        else:
            penalty = penalty.lower()

        if class_weight == "None" or class_weight is None:
            class_weight = None
        else:
            class_weight = class_weight.lower()

        model = LogisticRegression(
            penalty =penalty,
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state
        )

        params = {
            # 'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'model__C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }

        paramsAda = {
            'model__C': (0.0000001, 100.0),
            'model__solver': Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            # 'model__penalty': Categorical(['l1', 'l2', 'elasticnet', 'none']),

        }

        model_type = 'logistic_regression'
        return [model, paramsAda, params, model_type]

    def Vs_BaggingClassifier_pipeline(self,
                                      estimator=None,
                                      n_estimators=10,
                                      max_samples=1.0,
                                      max_features=1.0,
                                      bootstrap=True,
                                      random_state=None):
        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        if bootstrap == "True":
            bootstrap = True
        else:
            bootstrap = False

        model = BaggingClassifier(
            estimator=estimator_instance,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

        params = {
            'model__base_estimator': [SVC(), DecisionTreeClassifier()],
            'model__n_estimators': [3, 5, 10, 20, 50, 100]
        }

        paramsAda = {
            'model__base_estimator': Categorical([SVC(kernel='rbf'), DecisionTreeClassifier()]),
            'model__n_estimators': (2, 250),
        }

        model_type = 'bagging_classifier'
        return [model, paramsAda, params, model_type]

    def Vs_AdaBoostClassifier_pipeline(self,
                                       estimator=None,
                                       n_estimators=50,
                                       learning_rate=1.0,
                                       algorithm="SAMME.R",
                                       base_estimator=None,
                                       random_state=None):

        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        if base_estimator == "None" or base_estimator is None:
            base_estimator = None

        model = AdaBoostClassifier(
            estimator=estimator_instance,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            base_estimator=base_estimator,
            random_state=random_state
        )

        base_models = [
            SVC(kernel='rbf'),
            DecisionTreeClassifier()]
        params = {
            'model__base_estimator': base_models,
            'model__n_estimators': [10, 20, 50, 100, 200, 400],
            'model__learning_rate': range(0.0001, 10.0, 0.1),
            'model__loss': ['linear', 'square', 'exponential']
        }

        paramsAda = {
            'model__base_estimator': Categorical([SVC(kernel='rbf'), DecisionTreeClassifier()]),
            'model__n_estimators': (2, 250),
            'model__learning_rate': (0.0001, 10.0),
            'model__loss': Categorical(['linear', 'square', 'exponential']),
        }

        model_type = 'adaboost_classifier'
        return [model, paramsAda, params, model_type]

    def Vs_KNeighborsClassifier_pipeline(self,
                                         n_neighbors=5,
                                         weights="uniform",
                                         algorithm="auto",
                                         leaf_size=30,
                                         p=2,
                                         metric="minkowski"):

        if weights is not None:
            weights = weights.lower()

        algorithm = algorithm.lower()
        metric = metric.lower()

        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     weights=weights,
                                     algorithm=algorithm,
                                     leaf_size=leaf_size,
                                     p=p,
                                     metric=metric
                                     )

        params = {
            'model__n_neighbors': range(1, 51),
            'model__leaf_size': range(5, 65, 5),
            "model__metric": ["euclidean", "manhattan", "cityblock", "minkowski"],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        paramsAda = {
            'model__n_neighbors': (2, 51),
            'model__leaf_size': (5, 65),
            'model__metric': Categorical(["euclidean", "manhattan", "cityblock", "minkowski"]),
            'model__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
        }

        model_type = 'k_neighbors_classifier'
        return [model, paramsAda, params, model_type]

    def Vs_svmClassifier_Pipeline(self,
                                  C=1.0,
                                  kernel='rbf',
                                  degree=3,
                                  coef0=0.0,
                                  class_wight=None,
                                  gamma="scale",
                                  decision_function_shape="ovr"):
        if class_wight == "None" or class_wight is None:
            class_wight = None
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            coef0=coef0,
            class_weight=class_wight,
            gamma=gamma,
            decision_function_shape=decision_function_shape
        )
        params = {
            'model__c': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'model__degree': (1, 180),
            'model__ceofo': (0.0, 1.0),
            'model__cache_size': (100.0, 300.0),
            'model__class_weight': [[1, 12, 41, 42], 'balanced', None],
            'model__verbos': [True, False],
            'model__max_iter': (1, 50)
        }
        paramsAda = {
            'model__c': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
            'model__degree': range(1, 180),
            'model__ceofo': range(0.0, 1.0),
            'model__cache_size': range(100.0, 300.0),
            'model__class_weight': Categorical([[1, 12, 41, 42], 'balanced', None]),
            'model__verbos': Categorical([True, False]),
            'model__max_iter': range(1, 50, 3)
        }
        model_type = 'svm_classifier'
        return [model, model_type, params, paramsAda]

    def Vs_decision_tree_Pipline(self,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 class_weight=None,
                                 random_state=None):

        if max_depth == "None" or max_depth is None or max_depth == 0:
            max_depth = None
        if class_weight == "None" or class_weight is None:
            class_weight = None

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state
        )
        params = {
            'model__min_sample_spllit': range(1, 42, 2),
            'model__min_sample_leaf': range(0, 10)
        }
        paramsAda = {
            'model__min_sample_spllit': (1, 42),
            'model__min_sample_leaf': (0, 10)
        }
        model_type = 'decision_tree_classifier'
        return [model, model_type, params, paramsAda]

    def VS_naive_bayes_Pipeline(self):
        model = GaussianNB()

        return model

    def Vs_LogisticClassifier_Get_best_params(self,
                                              penalty="l2",
                                              C=1.0,
                                              class_weight=None,
                                              solver="lbfgs",
                                              max_iter=100,
                                              multi_class="auto",
                                              best_parameters=None,
                                              random_state=None
                                              ):

        solver = solver.lower()
        multi_class = multi_class.lower()

        if penalty == "None" or penalty is None:
            penalty = None
        else:
            penalty = penalty.lower()

        if class_weight == "None" or class_weight is None:
            class_weight = None
        else:
            class_weight = class_weight.lower()

        best_c = best_parameters['model__C']
        best_solver = best_parameters['model__solver']

        model = LogisticRegression(
            penalty=penalty,
            C=best_c,
            class_weight=class_weight,
            solver=best_solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state
        )

        return model

    def Vs_BaggingClassifier_Get_best_params(self,
                                             estimator=None,
                                             n_estimators=10,
                                             max_samples=1.0,
                                             max_features=1.0,
                                             bootstrap=True,
                                             best_parameters=None,
                                             random_state=None):

        # if estimator is not None:
        #     estimator = estimator.lower()

        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        if bootstrap == "True":
            bootstrap = True
        else:
            bootstrap = False

        best_base_estimator = best_parameters['model__base_estimator']
        best_n_estimators = best_parameters['model__n_estimators']

        model = BaggingClassifier(
            estimator=estimator_instance,
            n_estimators=best_n_estimators,
            base_estimator=best_base_estimator,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

        return model

    def Vs_AdaBoostClassifier_Get_best_params(self,
                                              estimator=None,
                                              n_estimators=50,
                                              learning_rate=1.0,
                                              algorithm="SAMME.R",
                                              base_estimator=None,
                                              best_parameters=None,
                                              random_state=None):

        estimator_mapping = {
            'SVC': SVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'None': None
        }

        estimator_class = estimator_mapping.get(estimator)

        # Create an instance of the estimator if it's not None
        estimator_instance = None
        if estimator_class is not None:
            estimator_instance = estimator_class()

        if base_estimator == "None" or base_estimator is None:
            base_estimator = None

        best_base_estimator = best_parameters.get('model__base_estimator', base_estimator)
        best_n_estimators = best_parameters.get('model__n_estimators', n_estimators)
        best_learning_rate = best_parameters.get('model__learning_rate', learning_rate)
        # Use algorithm parameter if model__loss is not in best_parameters
        best_loss = best_parameters.get('model__loss', algorithm)

        model = AdaBoostClassifier(
            estimator=estimator_instance,
            base_estimator=best_base_estimator,
            n_estimators=best_n_estimators,
            learning_rate=best_learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )

        return model

    def Vs_KNeighborsClassifier_Get_best_params(self,
                                                n_neighbors=5,
                                                weights="uniform",
                                                algorithm="auto",
                                                leaf_size=30,
                                                p=2,
                                                metric="minkowski",
                                                best_parameters=None):

        if weights is not None:
            weights = weights.lower()

        algorithm = algorithm.lower()
        metric = metric.lower()

        best_n_neighbors = best_parameters['model__n_neighbors']
        best_leaf_size = best_parameters['model__leaf_size']
        best_metric = best_parameters['model__metric']
        best_algorithm = best_parameters['model__algorithm']

        model = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                                     weights=weights,
                                     algorithm=best_algorithm,
                                     leaf_size=best_leaf_size,
                                     p=p,
                                     metric=best_metric
                                     )

        return model

    def Vs_decision_tree_Get_best_params(self,
                                         criterion='gini',
                                         max_depth=None,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         class_weight=None,
                                         best_parameters=None,
                                         random_state=None):
        if max_depth == "None" or max_depth is None or max_depth == 0:
            max_depth = None
        if class_weight == "None" or class_weight is None:
            class_weight = None

        best_min_sample_split = best_parameters['model__min_sample_spllit']
        best_min_sample_split_leaf = best_parameters['model__min_sample_leaf']

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=best_min_sample_split,
            min_samples_leaf=best_min_sample_split_leaf,
            class_weight=class_weight,
            random_state=random_state
        )
        return model

    def VS_naive_bayes_Get_best_params(self):
        model = GaussianNB()

        return model

    def Vs_svm_Get_best_params(self,
                               C=1.0,
                               kernel='rbf',
                               degree=3,
                               coef0=0.0,
                               # cache_size=200.0,
                               class_wight=None,
                               gamma="scale",
                               decision_function_shape="ovr",
                               # verbos=False,
                               # max_iter=1,
                               best_parameters=None
                               ):
        if class_wight == "None":
            class_wight = None
        # if verbos == "False":
        #     verbos = False
        # else:
        #     verbos = True

        best_c = best_parameters['model__c']
        best_kernel = best_parameters['model__kernel'],
        best_degree = best_parameters['model__degree'],
        best_coefo = best_parameters['model__ceofo'],
        best_cache_size = best_parameters['model__cache_size'],
        best_class_weight = best_parameters['model__class_weight'],
        best_verbos = best_parameters['model__verbos'],
        best_max_iter = best_parameters['model__max_iter'],

        model = SVC(
            C=best_c,
            kernel=best_kernel,
            degree=best_degree,
            coef0=best_coefo,
            # cache_size=best_cache_size,
            class_weight=best_class_weight,
            gamma=gamma,
            decision_function_shape=decision_function_shape,
            # verbose=best_verbos,
            # max_iter=best_max_iter
        )

        return model

    def Vs_decision_tree_classifier(self,
                                    criterion='gini',
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    class_weight=None,
                                    random_state=None):
        if max_depth == "None" or max_depth is None or max_depth == 0:
            max_depth = None

        if class_weight == "None" or class_weight is None:
            class_weight = None

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state
        )
        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()
        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_naive_bayes_classifier(self):
        model = GaussianNB()

        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()
        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}

    def Vs_svm_classifier(self,
                          C=1.0,
                          kernel='rbf',
                          degree=3,
                          coef0=0.0,
                          # cache_size=200.0,
                          class_wight=None,
                          gamma="scale",
                          decision_function_shape="ovr"
                          # verbos=False,
                          # max_iter=1,
                          # random_state=None
                          ):
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            coef0=coef0,
            # cache_size=cache_size,
            class_weight=class_wight,
            gamma=gamma,
            decision_function_shape=decision_function_shape
            # verbose=verbos,
            # max_iter=-max_iter,
            # random_state=random_state
        )

        self.model = model
        y_train_pred, y_val_pred, y_test_pred = self.Vs_fit_and_predict_model()
        train_score_dict, val_score_dict, test_score_dict = self.Vs_Score_cls_All_Data(y_train_pred, y_val_pred,
                                                                                       y_test_pred)

        return {'x_train': self.X_train, 'x_val': self.X_val, 'x_test': self.X_test,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
                'y_train': np.array(self.y_train), 'y_val': np.array(self.y_val), 'y_test': np.array(self.y_test),
                'train_score_dict': train_score_dict, 'val_score_dict': val_score_dict,
                'test_score_dict': test_score_dict, 'model_params': model.get_params()}
