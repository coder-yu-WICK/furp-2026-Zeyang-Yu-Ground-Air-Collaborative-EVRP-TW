# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 1 — 2026-6-11

**Attended this week's meeting:** Yes 

**Progress this week**

- Successfully configured a Python virtual environment and installed Google OR-Tools.

- Ran and analyzed the official TSP (Travelling Salesperson Problem) example.

- Studied the VRP (Vehicle Routing Problem) example and understood the overall workflow of OR-Tools routing optimization.

- Learned the roles of RoutingIndexManager and RoutingModel.

- Studied callback functions and understood how the solver retrieves distance and demand information.

- Learned the concept of Dimensions in OR-Tools, including Distance Dimension and Capacity Dimension.

- Compared VRP and CVRP and understood how vehicle capacity constraints are implemented using AddDimensionWithVehicleCapacity().

- Analyzed CVRP code and understood how customer demands and vehicle capacities are represented in optimization models.

- Built a conceptual hierarchy of routing optimization problems:

  TSP → VRP → CVRP → VRPTW → EVRP

- Developed a modular understanding of OR-Tools programs:

  Data → Manager → RoutingModel → Callback → Dimension → Constraint → Search → Solve → Print

- Connected OR-Tools examples with the SEP project topic on VRP, EVRP-TW, and truck-drone collaborative routing optimization.

**Challenges & blockers**

- Python fundamentals remain a bottleneck when reading OR-Tools source code.

- Although the overall optimization workflow is becoming clearer, independently writing routing models remains difficult.

- More practice is needed to understand OR-Tools APIs and solver internals.

- Need deeper understanding of how Dimensions and Constraints interact during optimization.

**Next steps**

- Continue studying CVRP examples and modify model parameters independently.

- Learn VRPTW (Vehicle Routing Problem with Time Windows).

- Explore Google Distance Matrix API and understand how real-world distance data can replace manually defined distance matrices.

- Strengthen understanding of OR-Tools architecture without relying solely on line-by-line code explanations.

- Begin studying EVRP-TW concepts and energy-related constraints.

**Hours spent (optional):**

- Approximately 12–15 hours.

**Links (optional):**

- Google OR-Tools Documentation

- TSP Example

- VRP Example

- CVRP Example
