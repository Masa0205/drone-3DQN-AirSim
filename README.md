# drone-3DQN-AirSim



## Overview

This project trains a drone tracking policy in a custom AirSim environment using the 3DQN (Dueling Double Deep Q-Network) algorithm.

## Architecture
### 1. AirSim Environment
- AirSim is a flight simulator developed by Microsoft, built on Unreal Engine.
- The environment consists 2 objects: a drone agent and an arUco marker cube.

### 2. 3DQN Agent
- **3DQN** (Dueling Double DQN) addresses some of the limitations of **DQN**.
  
  - **Dueling**: The agent estimetes **state value V(s)** and **advantage A(s,a)** separately to better distinguish between the overall value of a state and the value of each possible action.

  - **Double**:The agent uses Double Q-learning to reduce overestimation bias by separating the action selection and action evaluation processes.

## Demonstration

### The trained agent in AirSim environment
![無題のビデオ ‐ Clipchampで作成](https://github.com/user-attachments/assets/810ac8c8-0496-433f-afd4-44a3d3660cdc)
