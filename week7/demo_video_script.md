# Demo Video Script — FURP 2026
## Ground-Air Collaborative EVRP-TW: Hybrid Optimization for Truck-Drone Delivery

**Duration**: 6–7 minutes
**Format**: Screen recording with voiceover (English or Chinese)

---

## Segment 1: Problem Introduction (0:00–1:00)

**Visual**: Title slide with project name

**Script**:
"Last-mile delivery faces three converging challenges: time windows from customers,
battery constraints from electric vehicles, and the opportunity of drone-assisted
delivery. Our project addresses all three simultaneously — the Ground-Air
Collaborative Electric Vehicle Routing Problem with Time Windows.

The question we answer: How do charging strategies and truck-drone coordination
affect the feasibility and efficiency of EVRP-TW?"

**Visual**: Problem illustration — map with depot, customers, trucks, drones, charging stations
- Show `fig_route_2drone_RC101_50c.png`

---

## Segment 2: Methodology (1:00–2:30)

**Visual**: Pipeline flow diagram

**Script**:
"Our solution is a hybrid optimization pipeline with five stages:

1. **Construction**: We adaptively select between Clarke-Wright Savings for clustered
   instances and POMO neural construction for random or mixed distributions.
2. **Repair**: Earliest Due Date reordering within and between routes — this is the
   decisive component that takes us from 50% to 100% time-window feasibility.
3. **Drone Insertion**: Cross-route greedy insertion with per-truck limits of two
   simultaneous drones. A composite-score fallback rejects net-negative missions.
4. **EV Simulation**: Forward battery-state tracking with charging station insertion.
   We support both linear constant-rate and non-linear piecewise charging profiles.
5. **Synchronization**: A two-pass algorithm that models truck waiting at drone
   recovery nodes and propagates cascading delays through the entire route."

**Visual**: Show code structure — week7/ directory, key files

---

## Segment 3: Four-Model Ablation (2:30–3:30)

**Visual**: Model comparison diagram (A → B → C → D)

**Script**:
"We conduct a systematic four-model ablation study:

- **Model A** is truck-only with EDD repair — our baseline.
- **Model B** adds linear EV charging at a constant 1.0 kWh per minute.
- **Model C** introduces non-linear charging: 1.5 times faster at low battery,
  normal in the mid-range, and half-speed near full charge.
- **Model D** adds full launch-recovery synchronization with cascading delay
  propagation — when a truck waits for its drone, all subsequent deliveries
  are pushed later.

Each model builds incrementally on the previous one, letting us measure the
marginal contribution of each constraint family."

**Visual**: EV charging curve plot, sync timeline diagram

---

## Segment 4: Key Results (3:30–5:30)

**Visual**: fig1_comprehensive_comparison.png

**Script**:
"Now for the results. Three key findings:

**First — Scale and Feasibility.**
Across all 18 instances — 6 Solomon types at 50, 100, and 200 customers — our
method achieves 100% time-window feasibility. Classical metaheuristics like NSGA-II,
P-ACO, and IVND achieve zero percent. They find cheaper routes but with massive
time-window violations.

**Second — Drone Savings.**
Two drones per truck deliver average cost savings of 17.4% at smaller scales and
13.8% at 200 customers. On random-type tight-time-window instances, savings reach
over 40%. But we also found a structural limitation: clustered C-type instances at
200 customers are drone-unfriendly, and our composite-score fallback correctly
rejects all drone missions there.

**Third — Synchronization.**
When we properly model launch-recovery synchronization, 74% of instances require
non-zero truck waiting time — averaging 45 minutes. This waiting cascades into
additional tardiness, revealing that drone missions which appear 'free' under
simplified evaluation actually cause significant temporal disruption."

**Visual**: Show each figure as you discuss it

---

## Segment 5: EV & Failure Cases (5:30–6:30)

**Visual**: EV ablation table, failure case diagrams

**Script**:
"On the electric vehicle side, we found that at standard fleet parameters — 100
kilowatt-hour battery, eight trucks for 200 customers — EV constraints are non-binding.
This is actually good news for fleet operators: with adequate battery capacity, range
anxiety may be unfounded for urban delivery. But our stress tests at 25 to 30 kilowatt-hours
reveal the threshold where charging becomes necessary — 3 to 7 charging station visits
per instance.

We also documented five systematic failure cases, each mapping to a specific constraint:
battery starvation, time-window tightening, capacity exceeded, synchronization failure,
and charging station necessity."

---

## Segment 6: Conclusion (6:30–7:00)

**Visual**: Key takeaways bullet points

**Script**:
"To conclude: We have presented the first method that simultaneously addresses
VRPTW, EV charging, truck-drone collaboration, and launch-recovery synchronization
at 200-customer scale. Our hybrid pipeline achieves 100% time-window feasibility
with 13 to 17 percent drone cost savings.

The code is open-source and all experiments are reproducible with fixed random seeds.
Thank you."

**Visual**: GitHub link / contact info

---

## Recording Instructions

1. **Screen setup**: VS Code with week7/ files visible, terminal with results
2. **Record**: Use OBS Studio or QuickTime screen recording
3. **Resolution**: 1920×1080 (1080p)
4. **Audio**: Use external microphone for clear voiceover
5. **Edit**: Add title cards between segments, trim pauses

## Key Figures to Show (in order)

| Timestamp | Figure | File |
|-----------|--------|------|
| 0:30 | Route map example | `figures/fig_route_comparison_nd_vs_2d.png` |
| 2:00 | Pipeline diagram | (create simple diagram) |
| 3:30 | Method comparison | `figures/fig1_comprehensive_comparison.png` |
| 4:15 | Drone impact | `figures/fig2_drone_impact.png` |
| 4:45 | EV ablation | `figures/fig4_ev_ablation.png` |
| 5:15 | Sync results | (show terminal output from run_sync_study.py) |
| 5:45 | Route map panel | `figures/fig7_route_map_panel.png` |
| 6:15 | Gap heatmap | `figures/fig5_gap_heatmap.png` |
