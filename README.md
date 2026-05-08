# 🕹️ Maze Solver Challenge: AI Intelligent Agent
> **"Build a game agent that escapes faster and smarter."**

---

## 🎓 Academic Context
* **University:** El Sewedy University of Technology (EUT)
* **Course:** CET251 - Artificial Intelligence
* **Project ID:** Project 8 - Maze Solver Challenge
* **Level:** Second-Year, Data Science & AI Technology
* **Team Members:** Omar Ahmed Ramadan & Ahmed Mohamed Aldmrdash & Moaaz ABD Aljawwad Fouad



---

## 📝 Project Overview
This project involves designing and implementing an intelligent game agent capable of navigating complex maze environments. The agent is tasked with finding the most efficient path from a starting point to a goal while avoiding obstacles and predicting potential risks (traps/ghosts) using a Neural Network.

The project is built upon the **UC Berkeley Pacman AI Framework**, adapted to meet the specific requirements of the CET251 course project.

### 🌟 Key Features
* **Pathfinding:** Implementation of BFS, DFS, and A* search algorithms.
* **Risk Prediction:** A Neural Network add-on that predicts risky cells using `MLPClassifier`.
* **Performance Analysis:** Comprehensive comparison of runtime, path length, and node expansions.
* **Dynamic Environments:** Support for multiple layouts (Tiny, Medium, and Big mazes).

---

## 🧠 Core AI Concepts

### 1. Agents & Environments (PEAS Analysis)
* **Performance Measure:** Minimum path length, minimum node expansions, and success rate.
* **Environment:** A grid-based maze with walls, traps, and a goal.
* **Actuators:** Movement actions (North, South, East, West).
* **Sensors:** Local perception of surrounding walls, goal direction, and ghost/trap proximity.

### 2. Search Algorithms
We implemented and compared three fundamental search strategies in `search.py`:
* **Breadth-First Search (BFS):** Guarantees the shortest path in unweighted graphs.
* **Depth-First Search (DFS):** Explores deep into the maze before backtracking.
* **A*** **Search:** Uses the **Manhattan Distance** heuristic to guide the search efficiently toward the goal.

### 3. Neural Network Integration (Risk Prediction)
Located in `risk_prediction.py`, we integrated a **Multi-Layer Perceptron (MLP)** model:
* **Features:** Proximity to ghosts, distance to goal, and surrounding wall configuration.
* **Logic:** The `NnRadarAgent` uses the trained model to predict if a move is "High Risk" or "Safe" based on local environment features.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.12+
* NumPy
* Scikit-learn

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/AI-Project.git
   cd AI-Project
