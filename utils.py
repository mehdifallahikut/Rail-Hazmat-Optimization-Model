# utils.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore
from typing import List, Dict, Tuple
from dataclasses import dataclass
@dataclass(frozen=True)
class MultiMove:
    """
    Represents a group move of cars from one track to another.

    Attributes:
        cars (Tuple[str, ...]): A tuple of cars being moved.
        from_track (str): The name of the source track ('S' for Source, 'T' for Temp).
        to_track (str): The name of the destination track ('S', 'T', or 'D' for Departure).
    """
    __slots__ = ['cars', 'from_track', 'to_track']
    cars: Tuple[str, ...]
    from_track: str
    to_track: str

def print_shunting_sequence(block: str, initial_order: List[str], final_order: List[str], solution: List[Dict[str, str]]) -> None:
    print("\nShunting operations per block:")
    print("-" * 40)
    print(f"Block {block}:")
    print(f"Initial order: {initial_order}")
    print(f"Target order: {final_order}")
    print(f"\nBlock {block} shunting sequence:")
    if solution:
        source_track = initial_order.copy()
        temp_track = []
        departure_track = []
        for i, move in enumerate(solution, 1):
            print(f"Move {i}: Move {move['car']} from {move['from']} to {move['to']}")
            source = source_track if move['from'] == 'S' else temp_track
            car = source.pop()
            if move['to'] == 'S':
                source_track.append(car)
            elif move['to'] == 'T':
                temp_track.append(car)
            else:
                departure_track.append(car)
            print_state({"S": source_track, "T": temp_track, "D": departure_track})
    else:
        print("No solution found")
    print(f"Target order:\t {final_order}")
    print(f"Number of shunting operations: {len(solution)}")
    print("-" * 40)

def multi_print_shunting_sequence(block: str, initial_order: List[str], final_order: List[str], solution: List[Dict[str, str]]) -> None:
    print("\nShunting operations per block (Multi):")
    print("-" * 40)
    print(f"Block {block}:")
    print(f"Initial order: {initial_order}")
    print(f"Target order: {final_order}")
    print(f"\nBlock {block} shunting sequence:")
    if solution:
        source_track = list(initial_order)
        temp_track = []
        departure_track = []
        for i, move in enumerate(solution, 1):
            cars = move['cars']
            print(f"Move {i}: Move {cars} from {move['from']} to {move['to']}")
            source = source_track if move['from'] == 'S' else temp_track
            group = source[-len(cars):]
            assert group == cars
            del source[-len(cars):]
            if move['to'] == 'S':
                source_track.extend(cars)
            elif move['to'] == 'T':
                temp_track.extend(cars)
            else:
                departure_track.extend(cars)
            print_state({"S": source_track, "T": temp_track, "D": departure_track})
    else:
        print("No solution found")
    print(f"Target order:\t {final_order}")
    print(f"Number of shunting operations: {len(solution)}")
    print("-" * 40)



def print_state(state: dict) -> None:
    colors = {
        'A': Fore.LIGHTRED_EX,
        'B': Fore.LIGHTGREEN_EX,
        'C': Fore.LIGHTBLUE_EX,
        'D': Fore.LIGHTYELLOW_EX,
        'E': Fore.LIGHTMAGENTA_EX,
        'F': Fore.LIGHTWHITE_EX
    }
    def colored_track(track):
        return ' '.join(colors.get(car, '') + car + Fore.RESET for car in track)
    print("Source Track:\t", colored_track(state["S"]))
    print("Temp Track:\t\t", colored_track(state["T"]))
    print("Departure Track:", colored_track(state["D"]))
    print()
def print_state0(state: dict) -> None:
    print("Source Track:\t", state["S"])
    print("Temp Track:\t\t", state["T"])
    print("Departure Track:", state["D"])
    print()
def plot_results(permutations, risks, operations):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Block Permutations')
    ax1.set_ylabel('Risk Value', color=color)
    for i, risk in enumerate(risks):
        ax1.annotate(f'{risk:.4f}', (i, risk), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Shunting Operations', color=color)
    for i, op in enumerate(operations):
        ax2.annotate(f'{op}', (i, op), textcoords="offset points", xytext=(0,-15), ha='center')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.xticks(range(len(permutations)), [str(p) for p in permutations], rotation=45)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('Risk Values and Shunting Operations by Permutation')
    plt.tight_layout()
    plt.show()

def entropy_weighted_sum(risk_values, ops_values):
    risk = np.array(risk_values)
    ops = np.array(ops_values)
    epsilon = 1e-10
    risk_range = np.max(risk) - np.min(risk)
    ops_range = np.max(ops) - np.min(ops)
    if risk_range < epsilon:
        risk_norm = np.ones_like(risk) * 0.5
    else:
        risk_norm = 1 - (np.max(risk) - risk) / risk_range
    if ops_range < epsilon:
        ops_norm = np.ones_like(ops) * 0.5
    else:
        ops_norm = 1 - (np.max(ops) - ops) / ops_range
    risk_sum = np.sum(risk_norm) + epsilon
    ops_sum = np.sum(ops_norm) + epsilon
    risk_p = risk_norm / risk_sum
    ops_p = ops_norm / ops_sum
    k = 1.0 / np.log(len(risk) + epsilon)
    risk_entropy = -k * np.sum([p * np.log(p + epsilon) if p > 0 else 0 for p in risk_p])
    ops_entropy = -k * np.sum([p * np.log(p + epsilon) if p > 0 else 0 for p in ops_p])
    risk_div = 1 - risk_entropy
    ops_div = 1 - ops_entropy
    total_div = risk_div + ops_div
    risk_weight = risk_div / total_div
    ops_weight = ops_div / total_div
    final_scores = risk_weight * risk_norm + ops_weight * ops_norm
    return final_scores, risk_weight, ops_weight

def calculate_risk(P: List[float], permutation: Tuple[str, ...], final_order_by_block: Dict[str, List[str]]) -> float:
    train_order = []
    for block in permutation:
        train_order.extend(final_order_by_block[block])
    risk = 0.0
    for position, car in enumerate(train_order, start=1):
        if car != 'F':
            risk += P[position - 1]
    return risk
def print_Shunting_Moves(block: str, initial_order: List[str], final_order: List[str], solution: List[MultiMove]) -> None:
    print("\nShunting operations per block (Multi):")
    print("-" * 40)
    print(f"Block {block}:")
    print(f"Initial order: {initial_order}")
    print(f"Target order: {final_order}")
    print(f"\nBlock {block} shunting sequence:")
    if solution:
        source_track = list(initial_order)
        temp_track = []
        departure_track = []
        for i, move in enumerate(solution, 1):
            cars = move.cars
            print(f"Move {i}: Move {cars} from {move.from_track} to {move.to_track}")
            source = source_track if move.from_track == 'S' else temp_track
            group = source[-len(cars):]
            assert group == list(cars)
            del source[-len(cars):]
            if move.to_track == 'S':
                source_track.extend(cars)
            elif move.to_track == 'T':
                temp_track.extend(cars)
            else:
                departure_track.extend(cars)
            print_state({"S": source_track, "T": temp_track, "D": departure_track})
    else:
        print("No solution found")
    print(f"Target order:\t {final_order}")
    print(f"Number of shunting operations: {len(solution)}")
    print("-" * 40)

