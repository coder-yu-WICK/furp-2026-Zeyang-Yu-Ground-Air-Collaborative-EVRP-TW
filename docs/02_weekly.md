Week 2 — 2026-06-20

Attended this week’s meeting: Yes

Progress this week

* Read the paper Electric Truck-based Robot Delivery Problem with Nonlinear Charging and focused on understanding the OR component before the ALNS stage.
* Built an OR-Tools based UAV-Truck routing framework from scratch.
* Generated random benchmark instances with scales of 50, 100, and 200 customers.
* Implemented heterogeneous fleet modelling consisting of 1 truck and 3 UAVs.
* Added customer time window constraints.
* Added truck-only customer nodes that cannot be served by UAVs.
* Added charging station locations into the routing network.
* Implemented a nonlinear charging function to approximate battery charging behaviour.
* Added distance-based battery/range constraints for both truck and UAV vehicles.
* Added preliminary truck-UAV coordination logic using soft penalties.
* Successfully tested the model on 50, 100, and 200 customer instances.

Experimental Results：

============================================================
 UAV-TRUCK PROBLEM WITH NONLINEAR CHARGING
 OR-Tools Implementation (Advanced)
============================================================

############################################################
 TESTING SCALE: 50 CLIENTS
############################################################

============================================================
 UAV-TRUCK PROBLEM WITH NONLINEAR CHARGING
 Scale: 50 clients, 3 UAVs + 1 Truck
 Charging Stations: [0, 37, 28, 42]
============================================================

Solving with time limit: 15 seconds...

==================================================
 EXPERIMENT RECORD: SCALE = 50 CLIENTS
==================================================
Feasibility Status : SUCCESS (Feasible)
Runtime (Seconds)  : 15.00 s
Objective Value    : 576 (Total Economic Cost)

Vehicle Route Details:

-> Truck (卡车)
   路径: 0 -> 15 -> 31 -> 43 -> 37 -> 42 -> 19 -> 14 -> 12 -> 18 -> 49 -> 0
   总距离: 202 单位
   访问客户数: 10

-> UAV (无人机) 1
   路径: 0 -> 21 -> 22 -> 4 -> 38 -> 48 -> 3 -> 35 -> 7 -> 28 -> 30 -> 6 -> 25 -> 10 -> 33 -> 11 -> 1 -> 17 -> 8 -> 9 -> 26 -> 34 -> 40 -> 39 -> 32 -> 45 -> 41 -> 29 -> 5 -> 50 -> 36 -> 46 -> 44 -> 16 -> 2 -> 27 -> 23 -> 47 -> 13 -> 20 -> 24 -> 0
   总距离: 618 单位
   访问客户数: 40
   充电站停靠: [0, 28]

==================================================
SUMMARY STATISTICS:
==================================================
Total Clients Visited: 50 / 50
Unvisited Clients: 0

Nonlinear Charging Analysis:
- Charging Stations: [0, 37, 28, 42]
- Battery Capacity: 2000
- Charging Efficiency: 0.85
- Nonlinear Exponent: 1.5
==================================================


############################################################
 TESTING SCALE: 100 CLIENTS
############################################################

============================================================
 UAV-TRUCK PROBLEM WITH NONLINEAR CHARGING
 Scale: 100 clients, 3 UAVs + 1 Truck
 Charging Stations: [0, 27, 6, 76]
============================================================

Solving with time limit: 30 seconds...

==================================================
 EXPERIMENT RECORD: SCALE = 100 CLIENTS
==================================================
Feasibility Status : SUCCESS (Feasible)
Runtime (Seconds)  : 30.00 s
Objective Value    : 815 (Total Economic Cost)

Vehicle Route Details:

-> Truck (卡车)
   路径: 0 -> 70 -> 32 -> 100 -> 14 -> 89 -> 71 -> 75 -> 54 -> 2 -> 36 -> 5 -> 82 -> 22 -> 34 -> 84 -> 31 -> 4 -> 48 -> 68 -> 95 -> 11 -> 17 -> 52 -> 92 -> 80 -> 28 -> 30 -> 56 -> 1 -> 85 -> 77 -> 6 -> 97 -> 25 -> 43 -> 37 -> 0
   总距离: 416 单位
   访问客户数: 36

-> UAV (无人机) 1
   路径: 0 -> 41 -> 88 -> 47 -> 58 -> 13 -> 72 -> 16 -> 51 -> 55 -> 86 -> 94 -> 26 -> 74 -> 57 -> 40 -> 93 -> 78 -> 42 -> 76 -> 65 -> 39 -> 29 -> 98 -> 63 -> 81 -> 49 -> 66 -> 69 -> 19 -> 18 -> 23 -> 73 -> 20 -> 64 -> 46 -> 44 -> 79 -> 50 -> 27 -> 12 -> 99 -> 24 -> 15 -> 45 -> 61 -> 21 -> 91 -> 9 -> 3 -> 67 -> 59 -> 96 -> 8 -> 83 -> 7 -> 10 -> 33 -> 87 -> 60 -> 35 -> 90 -> 62 -> 38 -> 53 -> 0
   总距离: 696 单位
   访问客户数: 64
   充电站停靠: [0, 76, 27]

==================================================
SUMMARY STATISTICS:
==================================================
Total Clients Visited: 100 / 100
Unvisited Clients: 0

Nonlinear Charging Analysis:
- Charging Stations: [0, 27, 6, 76]
- Battery Capacity: 2000
- Charging Efficiency: 0.85
- Nonlinear Exponent: 1.5
==================================================


============================================================
 TESTING LARGER SCALE: 200 CLIENTS
============================================================
Note: Larger scale may take more time to solve

============================================================
 UAV-TRUCK PROBLEM WITH NONLINEAR CHARGING
 Scale: 200 clients, 3 UAVs + 1 Truck
 Charging Stations: [0, 151, 49, 101]
============================================================

Solving with time limit: 60 seconds...

==================================================
 EXPERIMENT RECORD: SCALE = 200 CLIENTS
==================================================
Feasibility Status : SUCCESS (Feasible)
Runtime (Seconds)  : 60.00 s
Objective Value    : 1205 (Total Economic Cost)

Vehicle Route Details:

-> Truck (卡车)
   路径: 0 -> 26 -> 177 -> 155 -> 124 -> 24 -> 121 -> 19 -> 103 -> 176 -> 137 -> 70 -> 143 -> 123 -> 82 -> 130 -> 193 -> 165 -> 169 -> 16 -> 71 -> 151 -> 200 -> 147 -> 44 -> 58 -> 64 -> 116 -> 164 -> 150 -> 129 -> 65 -> 131 -> 187 -> 97 -> 101 -> 188 -> 161 -> 163 -> 154 -> 67 -> 156 -> 126 -> 38 -> 186 -> 21 -> 53 -> 179 -> 0
   总距离: 482 单位
   访问客户数: 47

-> UAV (无人机) 2
   路径: 0 -> 41 -> 76 -> 190 -> 78 -> 175 -> 43 -> 37 -> 183 -> 94 -> 120 -> 69 -> 135 -> 49 -> 88 -> 29 -> 160 -> 74 -> 15 -> 42 -> 61 -> 102 -> 173 -> 114 -> 108 -> 60 -> 1 -> 33 -> 139 -> 52 -> 10 -> 28 -> 111 -> 96 -> 83 -> 171 -> 178 -> 115 -> 34 -> 40 -> 107 -> 45 -> 47 -> 36 -> 54 -> 55 -> 110 -> 86 -> 2 -> 149 -> 113 -> 46 -> 180 -> 51 -> 75 -> 112 -> 72 -> 20 -> 192 -> 118 -> 162 -> 122 -> 13 -> 145 -> 196 -> 125 -> 185 -> 104 -> 167 -> 153 -> 105 -> 5 -> 79 -> 50 -> 128 -> 146 -> 73 -> 89 -> 23 -> 132 -> 100 -> 18 -> 170 -> 57 -> 31 -> 22 -> 133 -> 84 -> 91 -> 95 -> 17 -> 85 -> 159 -> 30 -> 148 -> 80 -> 142 -> 136 -> 11 -> 191 -> 87 -> 77 -> 56 -> 189 -> 152 -> 9 -> 62 -> 197 -> 48 -> 184 -> 117 -> 6 -> 138 -> 92 -> 134 -> 7 -> 59 -> 68 -> 144 -> 25 -> 3 -> 90 -> 109 -> 127 -> 35 -> 195 -> 8 -> 198 -> 141 -> 194 -> 93 -> 172 -> 4 -> 158 -> 98 -> 39 -> 32 -> 119 -> 14 -> 12 -> 181 -> 199 -> 99 -> 182 -> 140 -> 168 -> 174 -> 27 -> 66 -> 81 -> 166 -> 106 -> 63 -> 157 -> 0
   总距离: 1306 单位
   访问客户数: 153
   充电站停靠: [0, 49]

==================================================
SUMMARY STATISTICS:
==================================================
Total Clients Visited: 200 / 200
Unvisited Clients: 0

Nonlinear Charging Analysis:
- Charging Stations: [0, 151, 49, 101]
- Battery Capacity: 2000
- Charging Efficiency: 0.85
- Nonlinear Exponent: 1.5
==================================================


50 Customers

* Feasible solution found
* Runtime: 15s
* Objective Value: 576
* All 50 customers served

100 Customers

* Feasible solution found
* Runtime: 30s
* Objective Value: 815
* All 100 customers served

200 Customers

* Feasible solution found
* Runtime: 60s
* Objective Value: 1205
* All 200 customers served

Challenges & blockers

* OR-Tools is effective for standard routing problems but becomes restrictive when attempting to model detailed truck-UAV synchronization.
* The nonlinear charging function has been implemented but is not yet fully integrated into the optimization process.
* Launch and recovery synchronization between truck and UAV operations remains simplified.
* Large-scale instances require increasingly relaxed constraints to maintain feasibility.

Next steps

* Begin exploring PyVRP as an alternative framework for future development.
* Compare OR-Tools and PyVRP modelling flexibility for EVRP-related problems.
* Investigate more realistic battery state tracking and charging mechanisms.
* Continue improving truck-UAV coordination constraints.
* Prepare baseline comparison material for the Week 2 methodology evaluation task.

Hours spent (optional): ~20 hours

Links (optional):

* OR-Tools UAV-Truck implementation code
* Experimental result logs (50 / 100 / 200 customer instances)
* Paper: Electric Truck-based Robot Delivery Problem with Nonlinear Charging
