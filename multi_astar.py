# multi_astar.py
"""
A module for solving the train shunting problem using a multi-move A* algorithm.
Refined move-generation: strict prioritization (direct-to-D by length, then semi-fast useful blockers,
then standard S<->T moves). Added debug flag for optional logging.
"""

# ---------------------------
# 1. Imports
# ---------------------------
import heapq
import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from line_profiler import profile
from data import Block, BlockDict, Car
from utils import print_Shunting_Moves
import numpy as np


# ---------------------------
# 2. Data Models
# ---------------------------
@dataclass(frozen=True)
class MultiMove:
    """Represents a group move of cars from one track to another."""
    __slots__ = ['cars', 'from_track', 'to_track']
    cars: Tuple[str, ...]
    from_track: str
    to_track: str


@dataclass(frozen=True, slots=True)
class MultiState:
    """Represents a problem state, capturing the arrangement of cars on all tracks."""
    source_track: Tuple[str, ...]
    temp_track: Tuple[str, ...]
    departure_track: Tuple[str, ...]
    parent: Optional['MultiState'] = field(default=None, hash=False, compare=False)
    move: Optional[MultiMove] = field(default=None, hash=False, compare=False)

    def make_move(self, move: MultiMove) -> 'MultiState':
        """Applies a move to the current state to generate a new state."""
        k = len(move.cars)
        tracks = {'S': self.source_track, 'T': self.temp_track, 'D': self.departure_track}
        tracks[move.from_track] = tracks[move.from_track][:-k]
        tracks[move.to_track] += move.cars
        return MultiState(
            source_track=tracks['S'],
            temp_track=tracks['T'],
            departure_track=tracks['D'],
            parent=self,
            move=move
        )

    def __lt__(self, other: 'MultiState') -> bool:
        # tie-breaker for heap: prefer state with longer departure (closer to goal)
        return len(self.departure_track) < len(other.departure_track)


# ---------------------------
# 3. Solver Class
# ---------------------------
class MultiAStar:
    """Implements the A* algorithm to find the shortest shunting sequence by moving groups of cars."""
    __slots__ = ['initial_order', 'initial_temp', 'target_order', 'max_group_size',
                 '_heuristic_cache', 'node_limit', 'allow_fallback', 'best_state']

    def __init__(self, initial_order: List[str], target_order: List[str],
                 initial_temp: List[str] = [], max_group_size: int = 5,
                 node_limit: int = 1000000, allow_fallback: bool = True):
        self.initial_order = tuple(initial_order)
        self.initial_temp = tuple(initial_temp)
        self.target_order = tuple(target_order)
        self.max_group_size = max_group_size
        self.node_limit = node_limit
        self.allow_fallback = allow_fallback
        self._heuristic_cache: Dict[int, int] = {}

    # ---------------- Heuristic components ----------------
    @staticmethod
    @lru_cache(maxsize=50000)
    def _count_misaligned_cars(track_tuple: Tuple[str, ...],
                               target_tuple: Tuple[str, ...],
                               dep_pos: int) -> int:
        if not track_tuple:
            return 0
        track_len = len(track_tuple)
        remaining_target_len = len(target_tuple) - dep_pos

        maxcheck = 50
        max_match_len = min(track_len, remaining_target_len, maxcheck)
        for match_size in range(max_match_len, 0, -1):
            if track_tuple[-match_size:] == target_tuple[dep_pos: dep_pos + match_size]:
                return track_len - match_size
        return track_len

    def count_misaligned_cars(self, track: Tuple[str, ...], dep_pos: int) -> int:
        return self._count_misaligned_cars(track, self.target_order, dep_pos)

    # ---------------- Next-state generation ----------------
    def _get_next_states(self, state: MultiState) -> List[MultiState]:
        """
        Strategy:
        1) Find all direct moves to D (fast_moves). If any, return them sorted by length desc.
        2) If none direct, find semi-fast moves (move small blockers to other track to unlock a suffix).
           Only accept semi-fast moves that are useful:
             - suffix exists somewhere in remaining_target
             - len_suffix > len_blockers (net gain > 0)
             - blockers length <= self.max_group_size
             - len_suffix <= max_possible_k (so suffix is actionable soon)
        3) Finally generate standard S<->T moves and sort them by heuristic.
        """
        next_states: List[MultiState] = []
        dep_pos = len(state.departure_track)
        remaining_target_len = len(self.target_order) - dep_pos
        max_possible_k = min(self.max_group_size, len(self.target_order) - dep_pos)
        # --- 1. Fast Path: direct correct group to D (prefer longest) ---
        fast_moves = []
        for track_name, track_content in [('S', state.source_track), ('T', state.temp_track)]:
            if not track_content:
                continue
            limit = min(len(track_content), max_possible_k)
            # prefer the longest suffix that matches
            for k in range(limit, 0, -1):
                suffix = track_content[-k:]
                target_prefix = self.target_order[dep_pos: dep_pos + k]
                if suffix == target_prefix:
                    # record (len, move, track_name) and break (only longest per track)
                    fast_moves.append((k, track_name, state.make_move(MultiMove(suffix, track_name, 'D'))))
                    break

        # If there are direct moves, return them sorted by length (longest first).
        if fast_moves:
            fast_moves.sort(key=lambda x: x[0], reverse=True)
            return [m for _, _, m in fast_moves]

        # --- 2. Semi-fast Path: move blockers to open future match (floating match heuristic) ---
        semi_fast_moves = []
        remaining_target = self.target_order[dep_pos:]
        remaining_target_len = len(remaining_target)

        for track_name, track_content in [('S', state.source_track), ('T', state.temp_track)]:
            if not track_content:
                continue
            other_track = 'T' if track_name == 'S' else 'S'

            # Try small offsets: move few blockers to open up useful suffix
            # CORRECTED LOGIC: We want to move 'blockers' (top) to expose 'buried_segment' (bottom).
            for offset in range(1, len(track_content)):
                buried_segment = track_content[:offset]  # Bottom of stack (to be exposed)
                blockers = track_content[offset:]  # Top of stack (to be moved)

                if not buried_segment:
                    continue

                # Check constraints on the 'buried' segment (is it worth it?)
                if len(buried_segment) > remaining_target_len:
                    continue
                if len(buried_segment) > max_possible_k:
                    continue

                # Check constraints on the 'blockers' (can we move them?)
                if len(blockers) > self.max_group_size:
                    continue

                # Net gain heuristic: exposed length > moved length
                if len(buried_segment) < len(blockers):
                    continue

                found_at = -1
                for start in range(remaining_target_len - len(buried_segment) + 1):
                    if tuple(buried_segment) == tuple(remaining_target[start:start + len(buried_segment)]):
                        found_at = start
                        break
                if found_at < 0:
                    continue

                net_gain = len(buried_segment) - len(blockers)
                # Prioritize larger exposed segments, then larger net gain, then smaller blockers
                score = (len(buried_segment), net_gain, -len(blockers))

                # Move the BLOCKERS, not the buried segment
                new_state = state.make_move(MultiMove(blockers, track_name, other_track))
                semi_fast_moves.append((score, new_state))

        if semi_fast_moves:
            semi_fast_moves.sort(key=lambda x: x[0], reverse=True)
            next_states.extend([s for _, s in semi_fast_moves])

        # --- 3. Standard moves (S <-> T) as fallback ---
        std_moves = []
        # generate S -> T moves
        for k in range(1, min(len(state.source_track), self.max_group_size) + 1):
            std_moves.append(state.make_move(MultiMove(state.source_track[-k:], 'S', 'T')))
        # generate T -> S moves
        for k in range(1, min(len(state.temp_track), self.max_group_size) + 1):
            std_moves.append(state.make_move(MultiMove(state.temp_track[-k:], 'T', 'S')))

        # sort the standard moves by heuristic (lower heuristic better)
        # std_moves.sort(key=lambda s: len(s.departure_track), reverse=True)

        next_states.extend(std_moves)
        return next_states

    # ---------------- Heuristic ----------------
    @lru_cache(maxsize=10000)
    def _heuristic_compute(self, state: MultiState) -> int:
        dep_len = len(state.departure_track)
        remaining = len(state.source_track) + len(state.temp_track)
        if remaining == 0:
            return 0

        base = (remaining + self.max_group_size - 1) // self.max_group_size
        misplaced_s = self.count_misaligned_cars(state.source_track, dep_len)
        misplaced_t = self.count_misaligned_cars(state.temp_track, dep_len)

        good_suffix = 0
        for a, b in zip(state.departure_track[::-1], self.target_order[::-1]):
            if a == b:
                good_suffix += 1
            else:
                break

        potential = 0
        for track in [state.source_track, state.temp_track]:
            for k in range(1, min(len(track), self.max_group_size) + 1):
                if track[-k:] == self.target_order[dep_len:dep_len + k]:
                    potential = max(potential, k)
                    break

        return base + (misplaced_s + misplaced_t) // 2 - good_suffix // 3 - potential

    def heuristic(self, state: MultiState) -> int:
        key = hash(state)
        if key not in self._heuristic_cache:
            self._heuristic_cache[key] = self._heuristic_compute(state)
        return self._heuristic_cache[key]

    # ---------------- Main Solve ----------------
    def solve(self) -> Tuple[List[MultiMove], bool, int, Optional[MultiState]]:
        """
        Solves the shunting problem using A* with multi-moves.
        Returns the path, optimality flag, max nodes expanded, and best_state if not optimal.
        """
        initial_state = MultiState(self.initial_order, self.initial_temp, ())
        self.best_state = initial_state  # Track the best state for fallback
        g_score: Dict[MultiState, int] = {initial_state: 0}
        tie_breaker = itertools.count()
        open_set = [(self.heuristic(initial_state), next(tie_breaker), initial_state)]
        closed: Dict[MultiState, int] = {}
        nodes_expanded = 0
        max_overall_node_expanded = 0

        while open_set:
            if nodes_expanded > max_overall_node_expanded:
                max_overall_node_expanded = nodes_expanded

            if nodes_expanded >= self.node_limit:
                if self.allow_fallback:
                    path_to_best = []
                    s = self.best_state
                    while s.parent and s.move:
                        path_to_best.append(s.move)
                        s = s.parent
                    path_to_best.reverse()
                    fallback_path = self._fallback_astar(self.best_state, self.max_group_size, self.node_limit)
                    return path_to_best + fallback_path, False, max_overall_node_expanded, None
                return [], False, max_overall_node_expanded, self.best_state

            _, _, current_state = heapq.heappop(open_set)
            if current_state in closed and g_score[current_state] >= closed[current_state]:
                continue

            closed[current_state] = g_score[current_state]

            if len(current_state.departure_track) > len(self.best_state.departure_track):
                self.best_state = current_state

            if current_state.departure_track == self.target_order:
                path = []
                s = current_state
                while s.parent and s.move:
                    path.append(s.move)
                    s = s.parent
                path.reverse()
                return path, True, max_overall_node_expanded, None

            current_g = g_score[current_state]
            for next_state in self._get_next_states(current_state):
                tentative_g = current_g + 1
                if tentative_g < g_score.get(next_state, float('inf')):
                    g_score[next_state] = tentative_g
                    f_new = tentative_g + self.heuristic(next_state)
                    heapq.heappush(open_set, (f_new, next(tie_breaker), next_state))
                nodes_expanded += 1

        if self.allow_fallback:
            path_to_best = []
            s = self.best_state
            while s.parent and s.move:
                path_to_best.append(s.move)
                s = s.parent
            path_to_best.reverse()
            fallback_path = self._fallback_astar(self.best_state, self.max_group_size, self.node_limit)
            return path_to_best + fallback_path, False, max_overall_node_expanded, None
        return [], False, max_overall_node_expanded, self.best_state

    # ---------------- Fallback ----------------
    def _fallback_astar(self, start_state: MultiState, current_fallback_size: int, node_limit: int) -> List[MultiMove]:
        """
        A recursive fallback solver that attempts to solve the remaining problem with a smaller group size.
        Preserves the path to the best state in each sub-search if it hits node_limit.
        """
        remaining_initial = list(start_state.source_track)
        remaining_temp = list(start_state.temp_track)
        remaining_target = list(self.target_order[len(start_state.departure_track):])
        num_remaining_cars = len(remaining_initial) + len(remaining_temp)
        if num_remaining_cars == 0:
            return []

        min_leftOver_car = max(len(remaining_initial), len(remaining_temp))
        effective_group_size = min(current_fallback_size, min_leftOver_car)
        new_node_limit = max(10000, node_limit * 0.95)  # Prevent negative values

        # Create a new solver for the subproblem
        fallback_solver = MultiAStar(
            initial_order=remaining_initial,
            initial_temp=remaining_temp,
            target_order=remaining_target,
            max_group_size=effective_group_size,
            node_limit=new_node_limit,
            allow_fallback=False  # Prevent infinite recursion in sub-solver
        )

        # Run the sub-solver, which now returns best_state if not optimal
        solution, is_optimal, _, sub_best_state = fallback_solver.solve()

        if is_optimal or solution:
            return solution
        else:
            # If failed, extract path to sub_best_state and recurse from there
            sub_path_to_best = []
            s = sub_best_state
            while s.parent and s.move:
                sub_path_to_best.append(s.move)
                s = s.parent
            sub_path_to_best.reverse()

            # Calculate next size
            next_size = 0
            if current_fallback_size > 100:
                next_size = max(6, current_fallback_size // 2)
            elif current_fallback_size > 50:
                next_size = max(6, current_fallback_size // 3)
            elif current_fallback_size > 25:
                next_size = max(6, current_fallback_size // 4)
            elif current_fallback_size > 12:
                next_size = max(6, current_fallback_size // 5)
            elif current_fallback_size > 6:
                next_size = max(6, current_fallback_size // 6)
            elif current_fallback_size > 1:
                next_size = current_fallback_size - 1

            if next_size > 0:
                # Recursive call from sub_best_state
                remaining_fallback_path = self._fallback_astar(sub_best_state, next_size, new_node_limit)
                return sub_path_to_best + remaining_fallback_path
            else:
                return []  # No solution after all fallbacks


# ---------------------------
# 4. Top-level Helper Function
# ---------------------------
def calculate_total_shunting_operations(permutation: Tuple[Block, ...],
                                        blocks: BlockDict,
                                        final_order_by_block: Dict[Block, List[Car]],
                                        print_flag: bool = False,
                                        node_limit: int = 10000) -> Tuple[int, bool, int]:
    max_node_expanded_overall = 0
    total_ops = 0
    solver = None
    is_overall_optimal = True

    for block in permutation:
        initial_order = blocks[block]
        node_limit = min(1_000_000, int((len(initial_order) * 10000) * (0.95 ** max(0, len(permutation) - 1))))
        final_order = final_order_by_block[block]
        solver = MultiAStar(initial_order, final_order, max_group_size=len(initial_order), node_limit=node_limit)
        solution, is_optimal, cur_node_expanded, best = solver.solve()

        if cur_node_expanded > max_node_expanded_overall:
            max_node_expanded_overall = cur_node_expanded
        if not is_optimal:
            is_overall_optimal = False
        if not solution:
            total_ops += 10000000
        if print_flag:
            print_Shunting_Moves(block, initial_order, final_order, solution)
        total_ops += len(solution)

    if solver:
        solver._count_misaligned_cars.cache_clear()
        solver._heuristic_cache.clear()

    return total_ops, is_overall_optimal, max_node_expanded_overall