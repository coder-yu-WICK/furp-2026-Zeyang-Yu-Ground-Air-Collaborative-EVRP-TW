# Week 2 – 20 June 2026

**Attended this week's meeting:** Yes

---

## Progress This Week

### 1. Literature Reproduction

This week focused on reproducing the optimization framework presented in the paper:

> **Electric Truck-Based Robot Delivery Problem with Nonlinear Charging**

The primary objective was to construct an operational prototype using OR-Tools and verify whether the main routing and charging components described in the paper could be reproduced within a constraint-programming framework.

---

### 2. OR-Tools Model Development

A UAV-Truck cooperative routing model was implemented using the OR-Tools Routing Solver.

#### Implemented Components

**Network Generation**
- Random customer locations generated in a two-dimensional space.
- Manhattan-distance matrix construction.
- Customer time-window generation.
- Depot initialization.

**Fleet Configuration**
- 1 electric truck.
- 3 UAVs.
- Heterogeneous vehicle routing structure.

**Routing Constraints**
- Vehicle-specific routing costs.
- Customer service assignment constraints.
- Truck-only customer restrictions.
- Customer dropping mechanism using penalty costs.
- Distance-based vehicle range limitations.

**Charging Infrastructure**
- Multiple charging stations generated within the network.
- Depot acts as the primary charging station.
- Additional charging stations selected from customer nodes.

**Battery Model**

A nonlinear charging function was implemented to approximate realistic battery behavior:

- Fast charging phase at low battery levels.
- Medium charging phase at moderate battery levels.
- Trickle charging phase at high battery levels.
- Charging efficiency parameter.
- Nonlinear charging exponent parameter.

**Optimization Framework**
- `RoutingIndexManager`
- `RoutingModel`
- Distance Dimension
- Time Dimension
- Guided Local Search (GLS) metaheuristic
- Multi-scale testing framework

---

### 3. Computational Experiments

Three benchmark instance sizes were tested.

| Instance Size | Fleet Configuration | Runtime | Objective Value | Feasible |
|--------------|---------------------|---------|-----------------|----------|
| 50 Customers | 1 Truck + 3 UAVs | 15 s | 576 | Yes |
| 100 Customers | 1 Truck + 3 UAVs | 30 s | 815 | Yes |
| 200 Customers | 1 Truck + 3 UAVs | 60 s | 1205 | Yes |

---

### 4. Experimental Results

#### Case 1: 50 Customers

**Performance**
- Feasible solution obtained.
- Runtime: 15 seconds.
- Objective value: 576.
- All customers successfully served.

**Vehicle Allocation**

| Vehicle | Distance Traveled | Customers Served |
|---------|-------------------|------------------|
| Truck | 202 units | 10 |
| UAV Fleet | 618 units | 40 |

**Charging Stations Visited:** Node 28

**Observation:** The optimization strongly favored UAV deployment, assigning approximately 80% of customers to UAV operations.

---

#### Case 2: 100 Customers

**Performance**
- Feasible solution obtained.
- Runtime: 30 seconds.
- Objective value: 815.
- All customers successfully served.

**Vehicle Allocation**

| Vehicle | Distance Traveled | Customers Served |
|---------|-------------------|------------------|
| Truck | 416 units | 36 |
| UAV Fleet | 696 units | 64 |

**Charging Stations Visited:** Nodes 76 and 27

**Observation:** The charging infrastructure began to play a visible role in route construction, with UAV routes naturally incorporating charging stops.

---

#### Case 3: 200 Customers

**Performance**
- Feasible solution obtained.
- Runtime: 60 seconds.
- Objective value: 1205.
- All customers successfully served.

**Vehicle Allocation**

| Vehicle | Distance Traveled | Customers Served |
|---------|-------------------|------------------|
| Truck | 482 units | 47 |
| UAV Fleet | 1306 units | 153 |

**Charging Stations Visited:** Node 49

**Observation:** The model remained computationally feasible at a larger scale and continued to assign most customers to UAV routes.

---

### 5. Analysis and Discussion

The experiments demonstrate that the implemented framework can successfully solve large-scale UAV-Truck routing instances while incorporating charging infrastructure and battery-related considerations.

Several important findings emerged:

1. The solver consistently allocated a majority of customers to UAV routes.
2. Charging stations were automatically integrated into UAV routing plans.
3. All tested instances achieved 100% customer coverage.
4. The framework remained computationally tractable up to 200 customers within the imposed time limits.
5. The current implementation successfully reproduces several core concepts described in the reference paper:
   - UAV-truck collaborative delivery.
   - Battery-aware routing.
   - Charging station integration.
   - Time-window constraints.
   - Large-scale vehicle routing optimization.

---

### 6. Current Limitations

Although the overall framework is operational, several aspects of the original formulation are not yet fully reproduced.

| Limitation | Description |
|------------|-------------|
| **Battery State Representation** | The nonlinear charging function exists as a computational component but is not currently embedded as an explicit decision variable within the optimization process. |
| **Launch and Recovery Synchronization** | Truck-UAV synchronization is approximated through routing logic rather than strict mathematical constraints. |
| **Charging Decisions** | Charging behavior is analyzed after route generation rather than optimized directly within the solver. |
| **Solver Flexibility** | The Routing Solver abstraction in OR-Tools limits the implementation of complex state-dependent constraints such as State-of-Charge transitions, nonlinear charging dynamics, and detailed truck-UAV synchronization decisions. |

---

### 7. Challenges & Blockers

The main challenge encountered this week was the mismatch between the complexity of the original mathematical formulation and the modeling flexibility offered by OR-Tools `RoutingModel`.

Many battery-related and synchronization-related constraints can only be approximated rather than represented exactly.

This limitation motivates investigation of alternative optimization frameworks.

---

### 8. Next Steps

The objectives for next week are:

1. Evaluate **PyVRP** as an alternative optimization framework.
2. Compare OR-Tools and PyVRP modeling flexibility.
3. Implement explicit battery-state tracking.
4. Develop truck-UAV launch and recovery synchronization mechanisms.
5. Integrate charging decisions directly into the optimization process.
6. Continue improving fidelity to the original mathematical model.

---

### 9. Hours Spent

**15–20 hours**

---

### 10. References

- [OR-Tools Vehicle Routing Solver Documentation](https://developers.google.com/optimization/routing)
- *Electric Truck-Based Robot Delivery Problem with Nonlinear Charging*
- Experimental source code and solver logs

---

**Repository:** [Link to your GitHub repo]

**Branch:** `week-2-progress`

**Status:** ✅ Completed
