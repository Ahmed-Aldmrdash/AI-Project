import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("--- AI Agent Risk Prediction Model ---")


np.random.seed(42)
num_samples = 1000

ghost_dist = np.random.randint(1, 11, num_samples)
goal_dist = np.random.randint(1, 21, num_samples)
walls_around = np.random.randint(0, 4, num_samples)


X = np.column_stack((ghost_dist, goal_dist, walls_around))


y = np.where((ghost_dist <= 2) | ((ghost_dist == 3) & (walls_around >= 2)), 1, 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Neural Network...")

model = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")


print("--- Testing Agent Moves ---")
test_cases = np.array([ 
    [1, 10, 2],  
    [8, 5, 0],  
    [3, 2, 3]    
])

predictions = model.predict(test_cases)

for i, state in enumerate(test_cases):
    status = "Danger ⚠️" if predictions[i] == 1 else "Safe ✅"
    print(f"State {i+1} -> Ghost Dist: {state[0]}, Goal Dist: {state[1]}, Walls: {state[2]} ===> Prediction: {status}")

