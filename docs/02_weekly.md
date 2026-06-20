Week 2 — 2026-06-20

Attended this week’s meeting: Yes

Progress this week

Literature Reproduction and OR-Tools Implementation

This week I focused on reproducing the optimization component of the paper Electric Truck-Based Robot Delivery Problem with Nonlinear Charging using OR-Tools.

The implementation currently includes:

* Random instance generator for large-scale UAV-Truck delivery scenarios.
* Manhattan-distance transportation network generation.
* Customer time-window constraints.
* Heterogeneous fleet consisting of:
    * 1 electric truck
    * 3 UAVs
* Truck-only customer constraints.
* Multiple charging station generation.
* Nonlinear charging model based on battery State-of-Charge (SOC).
* Vehicle-specific routing cost functions.
* Distance dimensions representing vehicle range limitations.
* Time dimensions representing routing and charging-related timing constraints.
* Customer dropping mechanism using disjunction penalties.
* Guided Local Search (GLS) metaheuristic for optimization.

Nonlinear Charging Model

A nonlinear charging function was implemented to approximate realistic battery charging behavior:

* Fast charging phase at low SOC.
* Moderate charging phase at medium SOC.
* Trickle charging phase at high SOC.
* Charging efficiency and nonlinear exponent included as model parameters.

The charging model is integrated into the routing framework and cost evaluation structure. Future work will incorporate charging-state transitions directly into the optimization constraints.

Computational Experiments

Experiments were conducted on three instance sizes.

Scale	Vehicles	Runtime	Objective	Feasible
50 Customers	1 Truck + 3 UAVs	15 s	576	Yes
100 Customers	1 Truck + 3 UAVs	30 s	815	Yes
200 Customers	1 Truck + 3 UAVs	60 s	1205	Yes

Key Results

50 Customers

* All 50 customers served.
* Truck served 10 customers.
* UAV served 40 customers.
* Charging station stop detected at Node 28.
* Objective value: 576.

100 Customers

* All 100 customers served.
* Truck served 36 customers.
* UAV served 64 customers.
* Charging station stops detected at Nodes 76 and 27.
* Objective value: 815.

200 Customers

* All 200 customers served.
* Truck served 47 customers.
* UAV served 153 customers.
* Charging station stop detected at Node 49.
* Objective value: 1205.

Observations

Several important behaviors emerged during the experiments:

1. The optimization consistently allocated a large proportion of customers to UAV routes.
2. Charging stations were naturally incorporated into UAV routes.
3. All tested instances achieved 100% customer coverage.
4. The model remained computationally feasible up to 200 customers within a one-minute time limit.
5. The implementation successfully reproduces several major structural components of the original paper:
    * UAV-truck cooperation
    * Battery-aware routing
    * Charging infrastructure
    * Time-window constraints
    * Large-scale routing optimization

At the same time, the implementation remains an approximation of the original formulation because charging decisions are not yet represented as explicit state transitions inside the solver.

Challenges & Blockers

Several limitations were identified during implementation:

* OR-Tools Routing Solver is not well suited for representing battery state transitions and nonlinear charging dynamics directly.
* Launch/recovery synchronization between truck and UAV is currently approximated rather than enforced exactly.
* Charging behavior is reflected through costs and post-analysis rather than exact optimization variables.
* The original paper uses a richer mathematical formulation than what can be naturally expressed within the RoutingModel framework.

Next Steps

Next week I plan to:

1. Investigate migration from OR-Tools Routing Solver to PyVRP.
2. Build a more flexible VRP framework with custom state variables.
3. Implement explicit UAV battery-state tracking.
4. Add launch/recovery synchronization constraints.
5. Compare OR-Tools and PyVRP in terms of flexibility and scalability.
6. Continue moving the implementation closer to the original mathematical formulation proposed in the paper.

Hours Spent

Approximately 15–20 hours.

Links

* OR-Tools implementation source code
* Experimental logs for 50 / 100 / 200 customer instances
* Reference paper: Electric Truck-Based Robot Delivery Problem with Nonlinear Charging
