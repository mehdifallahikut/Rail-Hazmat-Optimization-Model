# Rail-Hazmat-Optimization-Model
Research Project: Hazmat Routing and Car Placement in Railway Transportation using LBBD

Multi-Objective Optimization for Hazmat Rail Transportation
This repository contains the source code for my PhD research project. The goal is to optimize the routing and scheduling of hazardous materials (Hazmat) in rail networks, focusing on the trade-off between operational costs and safety risks.

Project Overview
In rail logistics, transporting Hazmat requires strict safety measures. This project implements a decision-support tool that determines:
The safest sequence of cars within a train (Permutation).
The optimal path through the rail network to minimize public exposure to risk.

Technical Approach
Optimization Model: I developed a Mixed-Integer Linear Programming (MILP) model that handles complex safety constraints, such as mandatory "buffer zones" between Hazmat cars.

Algorithm: Due to the complexity of the problem, I used Logic-Based Benders Decomposition (LBBD). This approach splits the problem into a Master Problem (sequencing) and a Sub-Problem (routing).
Risk Calculation: The model calculates risk based on population density (exposure) and derailment probabilities along specific rail segments.
