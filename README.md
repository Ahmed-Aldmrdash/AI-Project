# AI Maze Solver Challenge - Pacman Agent
**Course:** CET251 - Artificial Intelligence  
**Institution:** El Sewedy University of Technology (EUT)

## Project Overview
This project implements an intelligent game agent that navigates a maze with walls, avoiding traps/ghosts, and reaching the goal as fast as possible. The project compares different search algorithms in a game-like environment (Pacman Framework) and integrates a Machine Learning model to evaluate risks.

## Core AI Concepts Implemented
1. **Search Algorithms:** * Breadth-First Search (BFS)
   * Depth-First Search (DFS)
   * A* Search (using Manhattan Distance Heuristic)
2. **Neural Network Add-on (Risk Prediction):**
   * Uses `MLPClassifier` from `scikit-learn`.
   * The model predicts the probability of traps/ghosts in nearby cells based on local features (distance to ghost, distance to goal, and surrounding walls).

## Dependencies
Make sure you have Python installed along with the required libraries for the Neural Network:
```bash
pip install numpy scikit-learn
