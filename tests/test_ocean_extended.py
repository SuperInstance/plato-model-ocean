"""Extended tests for plato_model_ocean — Cell lifecycle, Ocean evolution, niches."""

import pytest
import torch
import numpy as np
from plato_model_ocean import Cell, Ocean, CELL_CONFIGS, make_task_stream, TASK_ROOMS


class TestCellExtended:
    def test_invalid_type(self):
        with pytest.raises(ValueError):
            Cell("nonexistent")

    def test_all_types(self):
        for cell_type in CELL_CONFIGS:
            cell = Cell(cell_type)
            assert cell.cell_type == cell_type
            assert cell.size > 0

    def test_forward(self):
        cell = Cell("sandbox", in_dim=8, out_dim=2)
        x = torch.randn(4, 8)
        out = cell(x)
        assert out.shape == (4, 2)

    def test_clone_mutated(self):
        cell = Cell("sandbox")
        child = cell.clone_mutated(scale=0.05)
        assert child.generation == 1
        assert child.parent_id == cell.genome_id

    def test_provenance(self):
        cell = Cell("sandbox", provenance=["room-1"])
        assert cell.provenance == ["room-1"]
        child = cell.clone_mutated()
        assert "room-1" in child.provenance


class TestMakeTaskStream:
    def test_drift(self):
        X, y = make_task_stream("drift", 50)
        assert X.shape == (50, 8)
        assert y.shape == (50,)

    def test_anomaly(self):
        X, y = make_task_stream("anomaly", 50)
        assert X.shape == (50, 8)

    def test_intent(self):
        X, y = make_task_stream("intent", 50)
        assert set(y.numpy()).issubset({0, 1, 2, 3})

    def test_priority(self):
        X, y = make_task_stream("priority", 50)
        assert set(y.numpy()).issubset({0, 1, 2})

    def test_relevance(self):
        X, y = make_task_stream("relevance", 50)
        assert set(y.numpy()).issubset({0, 1})

    def test_unknown_task(self):
        with pytest.raises(ValueError):
            make_task_stream("nonexistent")


class TestTaskRooms:
    def test_all_tasks_have_rooms(self):
        for task, room in TASK_ROOMS.items():
            assert isinstance(room, str)
            assert "room-" in room


class TestOceanExtended:
    def test_add_cells(self):
        ocean = Ocean()
        ocean.add(Cell("sandbox"))
        ocean.add(Cell("tide_pool"))
        assert len(ocean.cells) == 2

    def test_train_tick(self):
        ocean = Ocean()
        for _ in range(3):
            ocean.add(Cell("sandbox"))
        X, y = make_task_stream("drift", 50)
        ocean.train_tick(X, y, "drift")
        assert ocean.tick == 1

    def test_ocean_repr(self):
        ocean = Ocean()
        r = repr(ocean)
        assert isinstance(r, str)
