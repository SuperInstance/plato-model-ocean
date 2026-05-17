# plato-model-ocean

Cellular intelligence ecosystem — evolutionary neural networks organized into ecological niches.

## Architecture

The Model Ocean treats neural networks as living cells in an ecosystem with four niches:

| Niche | Cells | Params | Lifespan | Mutation | Role |
|-------|-------|--------|----------|----------|------|
| 🔬 Sandbox | 100-200 | ~100 | 30 ticks | 30% | Rapid exploration, throwaway experiments |
| 🌊 Tide Pool | 20-50 | ~300 | 200 ticks | 10% | Task-specialized, room-aligned |
| 🐟 School | 5-15 | ~850 | 1000 ticks | 5% | Self-organizing clusters, fleet coordination |
| 🐋 Whale | 1-3 | ~2700 | 5000 ticks | 1% | Deep reasoning, slow but powerful |

**Population cap:** 300 cells. **Total param budget:** ~100K (400KB) — fits on any device.

## Installation

```bash
pip install plato-model-ocean
```

## Quick Start

```python
from plato_model_ocean import Cell, Ocean, CELL_CONFIGS, make_task_stream

# Create ocean on CPU (use "cuda" for GPU)
ocean = Ocean(device="cpu")

# Colonize with different niches
for _ in range(50):
    ocean.add(Cell("sandbox"))
for _ in range(10):
    ocean.add(Cell("tide_pool"))
ocean.add(Cell("whale", provenance=["room-deep-reasoning"]))

# Evolve for 50 ticks
for tick in range(50):
    task = ["drift", "anomaly", "intent", "priority", "relevance"][tick % 5]
    X, y = make_task_stream(task, 100)
    ocean.train_tick(X, y, task)
    if (tick + 1) % 10 == 0:
        ocean.promote()

print(ocean.summary())
```

## Task Streams

Five built-in synthetic task generators modeling PLATO room patterns:

- **drift** — Binary stability detection (room-drift-detect)
- **anomaly** — Binary anomaly flagging (room-anomaly-flag)
- **intent** — 4-class intent classification (room-intent-classify)
- **priority** — 3-class priority ranking (room-priority-rank)
- **relevance** — Binary tile relevance (room-tile-relevance)

## Evolution

Cells evolve through:
1. **Training** — All cells train on the current task stream each tick
2. **Fitness** — Accuracy-based fitness evaluation
3. **Reproduction** — Cells with fitness > 0.75 may spawn mutated offspring
4. **Death** — Cells exceeding their lifespan are removed
5. **Promotion** — High-performing cells graduate to the next niche (sandbox → tide pool → school → whale)

## License

MIT
