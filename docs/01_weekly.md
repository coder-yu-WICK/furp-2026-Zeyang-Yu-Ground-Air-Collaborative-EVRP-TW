### Week 1 – 2026-06-13

**Attended this week's meeting:** Yes

**Progress this week**
- Set up the project repository and became familiar with the weekly reporting workflow.
- Started learning Google OR-Tools for vehicle routing optimization.
- Studied the basic Vehicle Routing Problem (VRP) example and understood the overall workflow:
  - create_data_model()
  - RoutingIndexManager
  - RoutingModel
  - Callback registration
  - Search parameters
  - Solution generation
- Learned how OR-Tools manages internal routing indices and actual node IDs:
  - manager.IndexToNode()
  - manager.NodeToIndex()
- Studied route traversal methods:
  - routing.Start()
  - routing.NextVar()
  - routing.IsEnd()
- Learned how route costs are calculated using:
  - routing.GetArcCostForVehicle()
- Studied Distance Dimensions and how OR-Tools accumulates route distances.
- Studied Capacitated Vehicle Routing Problem (CVRP):
  - demand_callback()
  - RegisterUnaryTransitCallback()
  - AddDimensionWithVehicleCapacity()
- Studied Pickup and Delivery Problem (PDP):
  - AddPickupAndDelivery()
  - Vehicle consistency constraints
  - Pickup-before-delivery constraints
- Started learning Vehicle Routing Problem with Time Windows (VRPTW):
  - Time callback
  - Time Dimension
  - Time window constraints
  - Waiting time concepts
- Built a conceptual understanding of how different VRP variants are modeled by adding new dimensions and constraints.

**Challenges & blockers**
- Python programming fundamentals are still a bottleneck when reading OR-Tools source examples.
- Understanding OR-Tools internal index mapping remains difficult.
- Still learning how callback functions interact with the solver internally.
- Need more practice understanding dimensions and cumulative variables.

**Next steps**
- Continue studying VRPTW examples.
- Learn EVRP modeling concepts and battery-related constraints.
- Practice modifying official OR-Tools examples independently.
- Build a small custom routing optimization example.
- Strengthen Python fundamentals, especially functions, classes, and callback mechanisms.

**Hours spent (optional):**
- 20–25 hours

**Links (optional):**
- Google OR-Tools Routing Documentation
- VRP Example
- CVRP Example
- PDP Example
- VRPTW Example
