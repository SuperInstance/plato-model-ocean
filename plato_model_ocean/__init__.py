"""
plato-model-ocean — Cellular Intelligence Ecosystem

The Model Ocean is an evolutionary ecosystem of neural network cells organized
into four niches by size and role:

    🔬 Sandboxes: 100-200 cells, tiny (~100 params), short-lived, high mutation
    🌊 Tide Pools: 20-50 cells, small (~300 params), task-specialized
    🐟 Schools: 5-15 cells, medium (~850 params), self-organizing clusters
    🐋 Whales: 1-3 cells, large (~2700 params), slow but deep

Usage:
    from plato_model_ocean import Cell, Ocean, CELL_CONFIGS, make_task_stream, TASK_ROOMS

    ocean = Ocean()
    ocean.add(Cell('sandbox'))
    X, y = make_task_stream('drift', 200)
    ocean.train_tick(X, y, 'drift')
    print(ocean.summary())
"""

import hashlib
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = [
    "Cell",
    "Ocean",
    "CELL_CONFIGS",
    "make_task_stream",
    "TASK_ROOMS",
    "MAX_POP",
]

# ---------------------------------------------------------------------------
# Cell architecture — parameterized by niche
# ---------------------------------------------------------------------------

CELL_CONFIGS = {
    "sandbox": {"h": 8, "life": 30, "mut": 0.3, "cost": 0.001},
    "tide_pool": {"h": 16, "life": 200, "mut": 0.1, "cost": 0.005},
    "school": {"h": 32, "life": 1000, "mut": 0.05, "cost": 0.01},
    "whale": {"h": 64, "life": 5000, "mut": 0.01, "cost": 0.05},
}

MAX_POP = 300

TASK_ROOMS = {
    "drift": "room-drift-detect",
    "anomaly": "room-anomaly-flag",
    "intent": "room-intent-classify",
    "priority": "room-priority-rank",
    "relevance": "room-tile-relevance",
}


class Cell(nn.Module):
    """A single neural-network cell in the ocean.

    Parameters
    ----------
    cell_type : str
        One of 'sandbox', 'tide_pool', 'school', 'whale'.
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output class dimension.
    provenance : list[str] | None
        Optional list of provenance tags (which PLATO rooms shaped this cell).
    """

    def __init__(self, cell_type: str, in_dim: int = 8, out_dim: int = 2, provenance=None):
        super().__init__()
        if cell_type not in CELL_CONFIGS:
            raise ValueError(f"Unknown cell type {cell_type!r}; choose from {list(CELL_CONFIGS)}")
        h = CELL_CONFIGS[cell_type]["h"]
        self.cell_type = cell_type
        self.config = CELL_CONFIGS[cell_type]
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )
        self.fitness: float = 0.0
        self.age: int = 0
        self.generation: int = 0
        self.genome_id: str = hashlib.md5(
            f"{cell_type}:{random.random()}".encode()
        ).hexdigest()[:8]
        self.parent_id: str | None = None
        self.provenance: list[str] = provenance or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clone_mutated(self, scale: float = 0.05) -> "Cell":
        """Return a mutated copy of this cell."""
        child = Cell(self.cell_type)
        child.load_state_dict(self.state_dict())
        child.generation = self.generation + 1
        child.parent_id = self.genome_id
        child.provenance = self.provenance.copy()
        with torch.no_grad():
            for p in child.parameters():
                if random.random() < self.config["mut"]:
                    p.add_(torch.randn_like(p) * scale)
        return child


# ---------------------------------------------------------------------------
# Data — multiple task streams from PLATO rooms
# ---------------------------------------------------------------------------

def make_task_stream(task: str, n: int = 200):
    """Generate a synthetic classification dataset for a given task.

    Returns (X, y) as FloatTensor / LongTensor on CPU.
    """
    X = np.zeros((n, 8))
    y = np.zeros(n, dtype=int)

    if task == "drift":
        for i in range(n):
            drift = np.random.exponential(0.3)
            X[i] = [
                np.random.random(), drift, np.random.random(),
                np.random.poisson(5), np.random.random(),
                np.random.random(), np.random.exponential(0.5), np.random.random(),
            ]
            y[i] = int(drift > 0.4)

    elif task == "anomaly":
        for i in range(n):
            is_anom = np.random.random() < 0.15
            X[i] = [
                np.random.random() * (5 if is_anom else 1),
                np.random.exponential(1), np.random.random(),
                np.random.random() * (3 if is_anom else 0.5),
                np.random.random(), np.random.random(), np.random.random(),
                np.random.exponential(0.3),
            ]
            y[i] = int(is_anom)

    elif task == "intent":
        for i in range(n):
            intent = np.random.randint(4)
            scales = [0.5, 1.5, 0.1, 2.0]
            X[i] = np.random.randn(8) * scales[intent]
            y[i] = intent

    elif task == "priority":
        for i in range(n):
            pri = np.random.randint(3)
            X[i] = [
                np.random.random(), np.random.random(), pri / 2,
                np.random.random(), np.random.random(), np.random.random(),
                np.random.random(), np.random.random(),
            ]
            y[i] = pri

    elif task == "relevance":
        for i in range(n):
            rel = np.random.random() < 0.6
            X[i] = [
                np.random.random() * (2 if rel else 0.5),
                np.random.random(), np.random.random() * (1.5 if rel else 0.3),
                np.random.random(), np.random.random(), np.random.random(),
                np.random.random(), np.random.random(),
            ]
            y[i] = int(rel)
    else:
        raise ValueError(f"Unknown task {task!r}; choose from {list(TASK_ROOMS)}")

    return torch.FloatTensor(X), torch.LongTensor(y)


# ---------------------------------------------------------------------------
# The Ocean
# ---------------------------------------------------------------------------

class Ocean:
    """Evolutionary ecosystem of :class:`Cell` instances.

    Cells compete for fitness, reproduce when successful, die of old age,
    and can be promoted from one niche to the next (sandbox → tide pool →
    school → whale).
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.cells: list[Cell] = []
        self.tick: int = 0
        self.device = torch.device(device)

    def add(self, cell: Cell) -> None:
        if len(self.cells) < MAX_POP:
            self.cells.append(cell.to(self.device))

    # ----- training --------------------------------------------------------

    def train_tick(self, X: torch.Tensor, y: torch.Tensor, task_name: str) -> None:
        """One evolutionary tick: train, evaluate, reproduce."""
        self.tick += 1
        X = X.to(self.device)
        y = y.to(self.device)

        new_cells: list[Cell] = []
        for cell in self.cells:
            cell.age += 1
            if cell.age > cell.config["life"]:
                continue  # die of old age

            opt = optim.Adam(cell.parameters(), lr=0.01)
            for _ in range(5):
                opt.zero_grad()
                out = cell(X)
                if out.shape[-1] != y.shape[-1]:
                    break
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()
                opt.step()

            with torch.no_grad():
                try:
                    pred = cell(X).argmax(1)
                    cell.fitness = (pred == y).float().mean().item()
                except Exception:
                    cell.fitness = 0.0

            new_cells.append(cell)

            # reproduction
            if cell.fitness > 0.75 and random.random() < cell.config["mut"] * 0.5:
                if len(new_cells) < MAX_POP:
                    child = cell.clone_mutated(0.05)
                    child.provenance.append(
                        f"{TASK_ROOMS.get(task_name, '?')}:tick{self.tick}"
                    )
                    new_cells.append(child)

        self.cells = new_cells

    # ----- niche promotion -------------------------------------------------

    def promote(self) -> None:
        """Promote successful cells to the next niche level."""
        promoted: list[tuple[int, Cell]] = []
        for i, cell in enumerate(self.cells):
            if cell.cell_type == "sandbox" and cell.fitness > 0.85 and cell.age > 8:
                p = Cell("tide_pool")
                p.provenance = cell.provenance + [
                    f"promoted:sandbox→tide_pool@tick{self.tick}"
                ]
                p.generation = cell.generation
                p.fitness = cell.fitness
                promoted.append((i, p))
            elif (
                cell.cell_type == "tide_pool"
                and cell.fitness > 0.9
                and cell.age > 50
            ):
                p = Cell("school")
                p.provenance = cell.provenance + [
                    f"promoted:tide_pool→school@tick{self.tick}"
                ]
                p.generation = cell.generation
                p.fitness = cell.fitness
                promoted.append((i, p))

        for idx, p in promoted:
            self.cells[idx] = p.to(self.device)

    # ----- census / summary -----------------------------------------------

    def census(self):
        counts: dict[str, int] = defaultdict(int)
        fit_sum: dict[str, float] = defaultdict(float)
        param_sum: dict[str, int] = defaultdict(int)
        best_fit: dict[str, float] = defaultdict(float)
        max_gen: dict[str, int] = defaultdict(int)

        for c in self.cells:
            counts[c.cell_type] += 1
            fit_sum[c.cell_type] += c.fitness
            param_sum[c.cell_type] += c.size
            best_fit[c.cell_type] = max(best_fit[c.cell_type], c.fitness)
            max_gen[c.cell_type] = max(max_gen[c.cell_type], c.generation)

        return counts, fit_sum, param_sum, best_fit, max_gen

    def summary(self) -> str:
        counts, fit_sum, param_sum, best_fit, max_gen = self.census()
        total = sum(counts.values())
        total_p = sum(param_sum.values())
        emoji = {
            "sandbox": "🔬",
            "tide_pool": "🌊",
            "school": "🐟",
            "whale": "🐋",
        }

        lines = [
            f"  Tick {self.tick:3d}: {total} cells | {total_p:,} params ({total_p * 4 / 1024:.1f}KB)"
        ]
        for ct in ["sandbox", "tide_pool", "school", "whale"]:
            if counts[ct] > 0:
                af = fit_sum[ct] / counts[ct]
                lines.append(
                    f"    {emoji[ct]} {ct:10s}: {counts[ct]:3d} cells | "
                    f"fit={af:.3f} best={best_fit[ct]:.3f} | "
                    f"gen={max_gen[ct]} | {param_sum[ct]:,} params"
                )
        return "\n".join(lines)
