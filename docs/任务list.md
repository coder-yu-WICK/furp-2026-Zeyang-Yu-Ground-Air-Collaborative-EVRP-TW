# Research Workflow for Ground-Air Collaborative EVRP-TW

> **Project Timeline:** Now – August 15  
> **Final Deliverables:** Poster · Final Report · Demo Video (5–8 min)

---

## 📌 Research Question

> **How do charging strategies and truck–drone coordination affect the feasibility and efficiency of Ground-Air Collaborative EVRP-TW?**

This single question drives all experiments, ablations, and analyses. The project is **not** about replicating many papers – it is about answering this question with clear evidence.

---

## 🧩 Project Architecture (Incremental Build)

| Model | Components |
|-------|------------|
| **Baseline** | Truck + UAV + Time Windows (TW) |
| **+ Charging** | Baseline + Linear Charging Stations |
| **+ Non‑linear** | Above + Non‑linear Charging Profile |
| **+ Synchronization** | Above + Launch‑Recovery Coordination |

### Ablation Study (4 models)

| Model ID | Description |
|----------|-------------|
| **A** | Baseline (no charging, no sync) |
| **B** | Baseline + Linear Charging |
| **C** | Baseline + Linear + Non‑linear Charging |
| **D** | Baseline + Linear + Non‑linear + Synchronization |

Compare across: **Objective value · Runtime · Feasibility**

---

## 🧪 Core Experiments

### Experiment 1 – Scale Test
- Instance sizes: **50, 100, 200** customers
- Metrics: Objective · Runtime · Feasible (%)
- Purpose: assess scalability

### Experiment 2 – Charging Study
- Compare: **No Charging** vs **Linear** vs **Non‑linear**
- Metrics: Objective · Charging times · Runtime
- Purpose: quantify the impact of charging models

### Experiment 3 – Synchronization Study
- Compare: **No Sync** (current) vs **Launch‑Recovery** (full sync)
- Metrics: UAV utilization · Makespan · Total cost
- Purpose: evaluate coordination benefit

---

## ❌ Failure Cases (≥3 required)

| Case | Setting | Result | Explanation |
|------|---------|--------|-------------|
| 1 | Battery capacity = 300 | **Infeasible** | UAV cannot return to charging station |
| 2 | Time windows tightened by 50% | **Infeasible** | Service windows impossible to meet |
| 3 | No charging stations (0) | **Infeasible** | UAV range insufficient for all deliveries |

These directly satisfy the supervisor's requirement for **at least 3 failure cases**.

---

## 📚 Reading List (Target: 5 deep, 10 skim)

### Phase 1 (Week 3) – Foundations
1. **VRPTW** – Time windows basics  
2. **EVRP** – Battery & charging constraints  
3. **Truck–Drone Routing** – Launch‑recovery operations  

### Phase 2 (Week 4) – Advanced
4. **Truck–Drone Synchronization** – coordination modelling  
5. **Non‑linear Charging** – realistic battery profiles *(already being implemented)*  

> **Total:** 5 deep reads + ~10 skimmed papers – enough for a solid literature foundation.

---

## 🗓️ 7‑Week Roadmap

| Week | Focus | Deliverables |
|------|-------|--------------|
| **3** (now) | Literature review | Summaries of 5 core papers |
| **4** | Stabilise Baseline | Run 50/100/200 instances (already nearly done) |
| **5** | Charging Module | Linear + Non‑linear; compare 50/100/200 |
| **6** | Synchronisation | Implement Launch‑Recovery; sync vs. no‑sync |
| **7** | Ablation Study | Run models A, B, C, D – collect metrics |
| **8** | Failure Cases | Generate 3 infeasible scenarios with analysis |
| **9** (early Aug) | Visualisation | Figures: route map, runtime, objective, feasibility |
| **10** (submission) | Finalise | Poster · Video · Final Report |

---

## 🎯 Immediate Priorities (do not get distracted)

- ❌ **No** PyVRP migration yet  
- ❌ **No** deep reinforcement learning / POMO  
- ✅ **Yes** – read the 5 core papers  
- ✅ **Yes** – implement Launch‑Recovery synchronisation  
- ✅ **Yes** – compare Linear vs. Non‑linear charging  
- ✅ **Yes** – run ablations + failure cases  

> If you complete these four, your poster will look like a **complete undergraduate optimisation research project** – not just a code replication exercise.

---

## 📎 Final Deliverables Checklist

- [ ] **Poster** – Problem · Baseline · Improvements · Experiments · Results · Conclusion  
- [ ] **Final Report** – What, why, and how well?  
- [ ] **Demo Video** (5–8 min) – code run, results, charts  

---

*Last updated: 2026-06-21*
