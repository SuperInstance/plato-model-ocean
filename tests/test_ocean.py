"""Tests for plato-model-ocean."""
import pytest
import torch

from plato_model_ocean import Cell, Ocean, CELL_CONFIGS, make_task_stream, TASK_ROOMS, MAX_POP


class TestCell:
    def test_cell_types(self):
        for ct in CELL_CONFIGS:
            c = Cell(ct)
            assert c.cell_type == ct
            assert c.size > 0

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            Cell("bogus")

    def test_forward_shape(self):
        c = Cell("sandbox", in_dim=8, out_dim=2)
        x = torch.randn(4, 8)
        out = c(x)
        assert out.shape == (4, 2)

    def test_clone_mutated(self):
        c = Cell("tide_pool")
        c.fitness = 0.9
        child = c.clone_mutated(0.05)
        assert child.generation == 1
        assert child.parent_id == c.genome_id
        assert child.cell_type == "tide_pool"

    def test_provenance(self):
        c = Cell("whale", provenance=["room-deep-reasoning"])
        assert c.provenance == ["room-deep-reasoning"]


class TestOcean:
    def test_add_cells(self):
        o = Ocean(device="cpu")
        for _ in range(10):
            o.add(Cell("sandbox"))
        assert len(o.cells) == 10

    def test_max_pop(self):
        o = Ocean(device="cpu")
        for _ in range(MAX_POP + 50):
            o.add(Cell("sandbox"))
        assert len(o.cells) == MAX_POP

    def test_train_tick(self):
        o = Ocean(device="cpu")
        for _ in range(5):
            o.add(Cell("tide_pool"))
        X, y = make_task_stream("drift", 50)
        o.train_tick(X, y, "drift")
        assert o.tick == 1
        assert len(o.cells) > 0

    def test_promote(self):
        o = Ocean(device="cpu")
        c = Cell("sandbox")
        c.fitness = 0.9
        c.age = 20
        o.cells = [c]
        o.promote()
        assert o.cells[0].cell_type == "tide_pool"

    def test_summary(self):
        o = Ocean(device="cpu")
        o.add(Cell("sandbox"))
        s = o.summary()
        assert "sandbox" in s

    def test_census(self):
        o = Ocean(device="cpu")
        o.add(Cell("sandbox"))
        o.add(Cell("school"))
        counts, *_ = o.census()
        assert counts["sandbox"] == 1
        assert counts["school"] == 1


class TestTaskStream:
    @pytest.mark.parametrize("task", list(TASK_ROOMS))
    def test_task_shapes(self, task):
        X, y = make_task_stream(task, 100)
        assert X.shape == (100, 8)
        assert y.shape == (100,)

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            make_task_stream("nonexistent", 10)
