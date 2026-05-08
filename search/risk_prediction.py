import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("--- AI Agent Risk Prediction Model ---")

# 1. توليد بيانات تدريب افتراضية (1000 حالة من بيئة المتاهة)
# Feature 1: المسافة لأقرب فخ/شبح (من 1 إلى 10 خطوات)
# Feature 2: المسافة للهدف (من 1 إلى 20 خطوة)
# Feature 3: عدد الحوائط المحيطة (من 0 إلى 3 حوائط)
np.random.seed(42)
num_samples = 1000

ghost_dist = np.random.randint(1, 11, num_samples)
goal_dist = np.random.randint(1, 21, num_samples)
walls_around = np.random.randint(0, 4, num_samples)

# دمج الخصائص في مصفوفة واحدة (X)
X = np.column_stack((ghost_dist, goal_dist, walls_around))

# 2. تحديد المنطق الخاص بالخطر (Labels - y)
# إذا كان الفخ على بُعد خطوتين أو أقل، وحوله حوائط تمنع الهرب، يعتبر "خطر مؤكد" (1)، وإلا "آمن" (0)
y = np.where((ghost_dist <= 2) | ((ghost_dist == 3) & (walls_around >= 2)), 1, 0)

# تقسيم البيانات إلى جزء للتدريب وجزء للاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. بناء وتدريب الشبكة العصبية (Neural Network)
print("Training Neural Network...")
# استخدمنا Multi-Layer Perceptron (MLP) بـ 2 Hidden Layers
model = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 4. اختبار الموديل وحساب الدقة
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# 5. تجربة الموديل على حالات جديدة (Testing with new hypothetical agent moves)
print("--- Testing Agent Moves ---")
test_cases = np.array([
    [1, 10, 2],  # الحالة الأولى: شبح قريب جداً، وحوائط كتير (مفترض خطر)
    [8, 5, 0],   # الحالة الثانية: شبح بعيد، ومفيش حوائط (مفترض آمن)
    [3, 2, 3]    # الحالة الثالثة: شبح على بُعد 3، بس محاصر بـ 3 حوائط (مفترض خطر)
])

predictions = model.predict(test_cases)

for i, state in enumerate(test_cases):
    status = "Danger ⚠️" if predictions[i] == 1 else "Safe ✅"
    print(f"State {i+1} -> Ghost Dist: {state[0]}, Goal Dist: {state[1]}, Walls: {state[2]} ===> Prediction: {status}")