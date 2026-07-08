# -*- coding: utf-8 -*-
"""
Fully vectorized POMO MDP Environment for EVRP-TW.
No Python loops — everything runs as tensor ops on CPU or MPS.

Truck routing with time windows + battery + capacity constraints.
Drone missions handled separately as post-processing.
"""

import torch
import math


class POMOEnv:
    """
    Vectorized environment for POMO truck routing.

    State: position, load, time, battery, visited mask
    Action: select next node (0 = depot, 1..N = customers)
    Reward: -(fixed_cost + distance_cost + tardiness_cost)

    All operations use pure tensor ops — no Python loops over batch/pomo dims.
    """

    def __init__(self, truck_speed=35.0, battery_capacity=100.0,
                 energy_per_km=1.0, tw_horizon=240.0,
                 fixed_cost=100.0, dist_cost_rate=2.0, tardiness_rate=1.0):
        self.truck_speed = truck_speed
        self.battery_capacity = battery_capacity
        self.energy_per_km = energy_per_km
        self.tw_horizon = tw_horizon
        self.fixed_cost = fixed_cost
        self.dist_cost_rate = dist_cost_rate
        self.tardiness_rate = tardiness_rate
        self.max_demand = 200.0
        self.max_steps_factor = 3  # max steps = problem_size * factor

    def load_problems(self, problems):
        """
        Load batch of problem instances. All tensors stay on CPU initially.

        Each problem dict has:
          depot_xy: (2,) tensor
          node_xy: (N, 2), node_demand: (N,), node_tw_start: (N,),
          node_tw_end: (N,), node_service: (N,)
        """
        self.batch_size = len(problems)
        self.problem_size = problems[0]['node_xy'].shape[0]
        self.pomo_size = self.problem_size

        # Stack all — normalize depot to (batch, 2)
        depots = []
        for p in problems:
            d = p['depot_xy']
            depots.append(d.reshape(-1)[:2])  # ensure (2,)
        self.depot_xy = torch.stack(depots)  # (batch, 2)

        self.node_xy = torch.stack([p['node_xy'] for p in problems])  # (batch, N, 2)
        self.node_demand = torch.stack([p['node_demand'] for p in problems])  # (batch, N)
        self.node_tw_start = torch.stack([p['node_tw_start'] for p in problems])
        self.node_tw_end = torch.stack([p['node_tw_end'] for p in problems])
        self.node_service = torch.stack([p['node_service'] for p in problems])

        # Distance matrix: (batch, N+1, N+1) — depot at index 0, customers at 1..N
        all_xy = torch.cat([self.depot_xy.unsqueeze(1), self.node_xy], dim=1)  # (batch, N+1, 2)
        diff = all_xy.unsqueeze(2) - all_xy.unsqueeze(1)  # (batch, N+1, N+1, 2)
        self.dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-10)

        # Travel time matrix: (batch, N+1, N+1)
        self.time_mat = self.dist_mat / self.truck_speed

        # Energy matrix: (batch, N+1, N+1)
        self.energy_mat = self.dist_mat * self.energy_per_km

        # Index helpers for gather ops
        b, pomo = self.batch_size, self.pomo_size
        self.BATCH_IDX = torch.arange(b, dtype=torch.long).unsqueeze(1).expand(b, pomo)
        self.POMO_IDX = torch.arange(pomo, dtype=torch.long).unsqueeze(0).expand(b, pomo)
        self.N_PLUS_1 = self.problem_size + 1

        # Pre-pad node attributes with zeros at index 0 (depot)
        # So we can gather with node indices directly
        zero_col = torch.zeros(b, 1)
        self._demand_padded = torch.cat([zero_col, self.node_demand], dim=1)  # (batch, N+1)
        self._tw_start_padded = torch.cat([zero_col, self.node_tw_start], dim=1)
        self._tw_end_padded = torch.cat([
            torch.full((b, 1), self.tw_horizon), self.node_tw_end  # depot TW is full horizon
        ], dim=1)
        self._service_padded = torch.cat([zero_col, self.node_service], dim=1)

    def reset(self, device='cpu'):
        """Reset all trajectories. Returns initial state dict."""
        b, pomo, N = self.batch_size, self.pomo_size, self.problem_size

        self.current_node = torch.zeros((b, pomo), dtype=torch.long, device=device)
        self.load = torch.ones((b, pomo), device=device)  # fraction of max_demand
        self.time = torch.zeros((b, pomo), device=device)
        self.battery = torch.ones((b, pomo), device=device)
        self.visited = torch.zeros((b, pomo, N + 1), device=device)  # 0=unvisited
        self.finished = torch.zeros((b, pomo), dtype=torch.bool, device=device)

        # Route storage: record (batch, pomo, max_steps) node indices
        max_steps = N * self.max_steps_factor + 2
        self.routes = torch.zeros((b, pomo, max_steps), dtype=torch.long, device=device)
        self.route_len = torch.zeros((b, pomo), dtype=torch.long, device=device)
        self._step_count = 0

        return self._get_state()

    def step(self, selected):
        """
        Execute one step for all trajectories.

        Args:
            selected: (batch, pomo) tensor of node indices

        Returns:
            done: (batch, pomo) bool
        """
        b, pomo = self.batch_size, self.pomo_size
        bidx, pidx = self.BATCH_IDX.to(selected.device), self.POMO_IDX.to(selected.device)

        # Record
        pos = self.route_len  # (batch, pomo)
        self.routes[bidx, pidx, pos] = selected
        self.route_len = pos + 1
        self._step_count += 1

        # Gather distances/times from current to selected
        cur = self.current_node  # (batch, pomo)
        dist_mat_dev = self.dist_mat.to(selected.device)
        time_mat_dev = self.time_mat.to(selected.device)
        energy_mat_dev = self.energy_mat.to(selected.device)

        dist = dist_mat_dev[bidx, cur, selected]  # (batch, pomo)
        travel_time = time_mat_dev[bidx, cur, selected]
        energy_used = energy_mat_dev[bidx, cur, selected]

        # Update time: arrival = current + travel
        arrival = self.time + travel_time

        # Gather TW info for selected node (vectorized)
        tw_start = self._tw_start_padded.to(selected.device)[bidx, selected]
        tw_end = self._tw_end_padded.to(selected.device)[bidx, selected]
        service = self._service_padded.to(selected.device)[bidx, selected]
        demand = self._demand_padded.to(selected.device)[bidx, selected]

        # Wait if early
        is_depot = (selected == 0)
        wait = torch.where(is_depot, torch.zeros_like(arrival),
                          torch.clamp(tw_start - arrival, min=0))
        self.time = arrival + wait + service

        # Update load: refill at depot
        self.load = torch.where(
            is_depot,
            torch.ones_like(self.load),
            self.load - demand / self.max_demand
        )

        # Update battery: recharge at depot
        self.battery = torch.where(
            is_depot,
            torch.ones_like(self.battery),
            self.battery - energy_used / self.battery_capacity
        )

        # Mark visited
        self.visited[bidx, pidx, selected] = 1
        # Depot always available
        self.visited[bidx, pidx, 0] = 0

        # Update position
        self.current_node = selected

        # Check completion: all customers visited?
        visited_count = self.visited.sum(dim=2)  # (batch, pomo), count of visited nodes
        all_done = visited_count >= self.problem_size
        self.finished = all_done

        # Safety: force-finish if max steps reached
        max_steps = self.problem_size * self.max_steps_factor + 2
        if self._step_count >= max_steps:
            self.finished[:] = True

        return self.finished.clone()

    def get_reward(self):
        """Calculate reward = -(fixed_cost + distance_cost + tardiness)."""
        b, pomo = self.batch_size, self.pomo_size
        device = self.routes.device
        bidx = self.BATCH_IDX.to(device)
        pidx = self.POMO_IDX.to(device)
        dist_mat_dev = self.dist_mat.to(device)
        tw_end_dev = self._tw_end_padded.to(device)
        tw_start_dev = self._tw_start_padded.to(device)
        service_dev = self._service_padded.to(device)

        max_len = int(self.route_len.max().item())
        if max_len < 2:
            return -torch.full((b, pomo), self.fixed_cost, device=device)

        routes = self.routes[:, :, :max_len]  # (batch, pomo, max_len)
        valid = torch.arange(max_len, device=device).unsqueeze(0).unsqueeze(0) < self.route_len.unsqueeze(-1)
        valid = valid.float()

        # Distance cost
        total_dist = torch.zeros((b, pomo), device=device)
        cur_time = torch.zeros((b, pomo), device=device)
        total_tardiness = torch.zeros((b, pomo), device=device)

        for step in range(1, max_len):
            prev = routes[:, :, step - 1]
            curr = routes[:, :, step]
            step_valid = valid[:, :, step]  # (batch, pomo)

            d = dist_mat_dev[bidx, prev, curr]
            total_dist = total_dist + step_valid * d

            # Time tracking
            arrival = cur_time + d / self.truck_speed
            tw_s = tw_start_dev[bidx, curr]
            tw_e = tw_end_dev[bidx, curr]

            wait = torch.where(curr > 0, torch.clamp(tw_s - arrival, min=0),
                              torch.zeros_like(arrival))
            arrival = arrival + wait
            tardy = torch.where(curr > 0, torch.clamp(arrival - tw_e, min=0),
                               torch.zeros_like(arrival))
            total_tardiness = total_tardiness + step_valid * tardy

            svc = service_dev[bidx, curr]
            cur_time = arrival + svc

            # Reset time when at depot
            cur_time = torch.where(curr == 0, torch.zeros_like(cur_time), cur_time)

        cost = (self.fixed_cost +
                total_dist * self.dist_cost_rate +
                total_tardiness * self.tardiness_rate)

        return -cost  # reward = negative cost

    def _get_state(self):
        """Build state dict with mask for model input."""
        return {
            'last_node_idx': self.current_node,
            'load': self.load,
            'time': self.time,
            'battery': self.battery,
            'ninf_mask': self._build_mask(),
        }

    def _build_mask(self):
        """
        Vectorized feasibility mask.

        Returns (batch, pomo, N+1) with 0=valid, -inf=masked.
        Masks: visited, capacity-infeasible, TW-infeasible, battery-infeasible.
        """
        b, pomo, N = self.batch_size, self.pomo_size, self.problem_size
        device = self.current_node.device

        # Move needed tensors to device
        dist_mat = self.dist_mat.to(device)
        time_mat = self.time_mat.to(device)
        energy_mat = self.energy_mat.to(device)
        demand_pad = self._demand_padded.to(device)
        tw_end_pad = self._tw_end_padded.to(device)
        bidx = self.BATCH_IDX.to(device)
        pidx = self.POMO_IDX.to(device)

        # Start with visited mask
        visited_mask = torch.where(
            self.visited > 0,
            torch.tensor(float('-inf'), device=device),
            torch.tensor(0.0, device=device)
        )  # (batch, pomo, N+1)
        # Depot always unmasked
        visited_mask[:, :, 0] = 0.0

        # For each candidate node j (1..N), check feasibility
        # Expand current state to (batch, pomo, N+1)
        cur = self.current_node  # (batch, pomo)

        # Distance from current node to all nodes: (batch, pomo, N+1)
        d_cur = dist_mat[bidx, cur, :]

        travel_time = d_cur / self.truck_speed
        arrival = self.time.unsqueeze(-1) + travel_time

        # TW check: can't arrive too late
        tw_infeasible = arrival > tw_end_pad.unsqueeze(1) + 60  # 60 slack

        # Capacity check
        cur_load = self.load * self.max_demand  # (batch, pomo)
        cap_infeasible = demand_pad.unsqueeze(1) > cur_load.unsqueeze(-1)

        # Battery check: need enough to reach node + return to depot
        d_j_to_depot = dist_mat[:, 0, :].unsqueeze(1)  # (batch, 1, N+1)
        energy_needed = (d_cur + d_j_to_depot) * self.energy_per_km
        cur_battery = self.battery * self.battery_capacity  # (batch, pomo)
        batt_infeasible = energy_needed > cur_battery.unsqueeze(-1) + 1e-6

        # Combine masks
        infeasible = tw_infeasible | cap_infeasible | batt_infeasible

        mask = torch.where(
            infeasible,
            torch.tensor(float('-inf'), device=device),
            visited_mask
        )

        # Depot always available
        mask[:, :, 0] = 0.0

        # Finished trajectories: only depot available
        mask = torch.where(
            self.finished.unsqueeze(-1).expand(-1, -1, N + 1),
            torch.where(
                torch.arange(N + 1, device=device).unsqueeze(0).unsqueeze(0) == 0,
                torch.tensor(0.0, device=device),
                torch.tensor(float('-inf'), device=device)
            ),
            mask
        )

        return mask

    def to_device(self, device):
        """Move persistent tensors to device."""
        self.current_node = self.current_node.to(device)
        self.load = self.load.to(device)
        self.time = self.time.to(device)
        self.battery = self.battery.to(device)
        self.visited = self.visited.to(device)
        self.finished = self.finished.to(device)
        self.routes = self.routes.to(device)
        self.route_len = self.route_len.to(device)
        self.BATCH_IDX = self.BATCH_IDX.to(device)
        self.POMO_IDX = self.POMO_IDX.to(device)
