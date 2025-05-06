# # # Define the objective function for PSO (feature selection)
# import numpy
# import sklearn.ensemble
# import sklearn.metrics
# from pyswarm import pso

# def objective_function(params):
#     # Convert continuous PSO output to binary (0 or 1) for feature selection
#     selected_features = numpy.where(params > 0.5, 1, 0).astype(int)  # Features selected if param > 0.5
    
#     # Select the features based on the current particle's vector
#     X_train_selected = DFTrainInputs.iloc[:, selected_features == 1]
#     X_test_selected = DFTestInputs.iloc[:, selected_features == 1]
    
#     # Train a Random Forest Classifier on the selected features
#     model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train_selected, DFTrainOutput)
#     y_pred = model.predict(X_test_selected)
    
#     # Calculate accuracy (you can use other metrics like F1, ROC, etc.)
#     accuracy = sklearn.metrics.accuracy_score(DFTestOutput, y_pred)
    
#     return -accuracy  # PSO minimizes, so return negative accuracy
# #  Define bounds for PSO (we'll use 0 and 1 to represent feature selection)
# lb = numpy.zeros(DFOriginalInputs.shape[1])  # Lower bounds (exclude all features initially)
# ub = numpy.ones(DFOriginalInputs.shape[1])   # Upper bounds (include all features initially)
# #  Apply PSO
# best_features, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=2)
# #  Get selected features based on the best particle found
# selected_features = numpy.where(best_features > 0.5, 1, 0).astype(int)
# # Display the selected feature indices
# print("Selected feature indices:", numpy.where(selected_features == 1)[0])
# # Train and evaluate the model with the selected features
# X_train_selected = DFTrainInputs.iloc[:, selected_features == 1]
# X_test_selected = DFTestInputs.iloc[:, selected_features == 1]
# model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_selected, DFTrainOutput)
# y_pred = model.predict(X_test_selected)
# accuracy = sklearn.metrics.accuracy_score(DFTestOutput, y_pred)
# print(f'Model accuracy with selected features: {accuracy}')

# # for different swarm size

# #  Apply PSO
# best_features, _ = pso(objective_function, lb, ub, swarmsize=5, maxiter=2)
# #  Get selected features based on the best particle found
# selected_features = numpy.where(best_features > 0.5, 1, 0).astype(int)
# # Display the selected feature indices
# print("Selected feature indices:", numpy.where(selected_features == 1)[0])
# #  Train and evaluate the model with the selected features
# X_train_selected = DFTrainInputs.iloc[:, selected_features == 1]
# X_test_selected = DFTestInputs.iloc[:, selected_features == 1]
# model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_selected, DFTrainOutput)
# y_pred = model.predict(X_test_selected)
# accuracy = sklearn.metrics.accuracy_score(DFTestOutput, y_pred)
# print(f'Model accuracy with selected features: {accuracy}')