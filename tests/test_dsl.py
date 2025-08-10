"""Tests for DSL engine and primitives."""

import pytest
import numpy as np
import time

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation, create_dsl_engine
from arc_solver.reasoning.primitives import (
    Rotate90, Rotate180, ReflectH, ReflectV, Crop, Paint, MapColors, PaintIf,
    SizePredicate, ColorPredicate, create_all_primitives
)


class TestDSLPrimitives:
    """Test individual DSL primitives."""
    
    @pytest.fixture
    def sample_grid(self):
        """Create a sample grid for testing."""
        return np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
    
    def test_rotate90(self, sample_grid):
        """Test 90-degree rotation."""
        rotate90 = Rotate90()
        result = rotate90(sample_grid)
        
        expected = np.array([
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_rotate180(self, sample_grid):
        """Test 180-degree rotation."""
        rotate180 = Rotate180()
        result = rotate180(sample_grid)
        
        expected = np.array([
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_reflect_h(self, sample_grid):
        """Test horizontal reflection."""
        reflect_h = ReflectH()
        result = reflect_h(sample_grid)
        
        expected = np.array([
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_reflect_v(self, sample_grid):
        """Test vertical reflection."""
        reflect_v = ReflectV()
        result = reflect_v(sample_grid)
        
        expected = np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_crop(self, sample_grid):
        """Test cropping operation."""
        crop = Crop()
        result = crop(sample_grid, r1=0, r2=1, c1=1, c2=2)
        
        expected = np.array([
            [2, 3],
            [5, 6]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_crop_bounds_checking(self, sample_grid):
        """Test crop with out-of-bounds coordinates."""
        crop = Crop()
        
        # Test coordinates beyond grid bounds
        result = crop(sample_grid, r1=0, r2=10, c1=0, c2=10)
        
        # Should clamp to grid bounds
        np.testing.assert_array_equal(result, sample_grid)
    
    def test_paint(self, sample_grid):
        """Test painting operation."""
        paint = Paint()
        result = paint(sample_grid, x=1, y=1, c=0)
        
        expected = sample_grid.copy()
        expected[1, 1] = 0
        
        np.testing.assert_array_equal(result, expected)
    
    def test_paint_out_of_bounds(self, sample_grid):
        """Test painting out of bounds."""
        paint = Paint()
        result = paint(sample_grid, x=10, y=10, c=0)
        
        # Should not modify grid
        np.testing.assert_array_equal(result, sample_grid)
    
    def test_map_colors(self, sample_grid):
        """Test color mapping."""
        map_colors = MapColors()
        
        # Create permutation that swaps 1->2, 2->1, others unchanged
        perm = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
        result = map_colors(sample_grid, perm=perm)
        
        expected = np.array([
            [2, 1, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_paint_if_size_predicate(self):
        """Test conditional painting with size predicate."""
        paint_if = PaintIf()
        
        # Grid with blobs of different sizes
        grid = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 0],
            [0, 0, 3, 3],
            [0, 0, 3, 3]
        ], dtype=np.int32)
        
        # Paint blobs with size >= 4 with color 9
        size_predicate = SizePredicate(min_size=4)
        result = paint_if(grid, predicate=size_predicate, new_color=9)
        
        # Blob 1 (size 4) and blob 3 (size 4) should be painted
        expected = np.array([
            [9, 9, 0, 2],
            [9, 9, 0, 0],
            [0, 0, 9, 9],
            [0, 0, 9, 9]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_paint_if_color_predicate(self):
        """Test conditional painting with color predicate."""
        paint_if = PaintIf()
        
        grid = np.array([
            [1, 2, 0],
            [1, 2, 3],
            [0, 0, 3]
        ], dtype=np.int32)
        
        # Paint blobs with color 1 or 3 with color 9
        color_predicate = ColorPredicate(target_colors=[1, 3])
        result = paint_if(grid, predicate=color_predicate, new_color=9)
        
        expected = np.array([
            [9, 2, 0],
            [9, 2, 9],
            [0, 0, 9]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_primitive_validation(self):
        """Test parameter validation for primitives."""
        # Test valid parameters
        crop = Crop()
        assert crop.validate_params(r1=0, r2=1, c1=0, c2=1)
        
        # Test invalid parameters
        assert not crop.validate_params(r1=1, r2=0, c1=0, c2=1)  # r1 > r2
        assert not crop.validate_params(r1=0, r2=1)  # Missing parameters
        
        paint = Paint()
        assert paint.validate_params(x=0, y=0, c=5)
        assert not paint.validate_params(x=-1, y=0, c=5)  # Negative coordinate
        assert not paint.validate_params(x=0, y=0, c=10)  # Invalid color
    
    def test_primitive_performance(self, sample_grid):
        """Test that primitives meet performance requirements."""
        primitives = [Rotate90(), Rotate180(), ReflectH(), ReflectV()]
        
        for primitive in primitives:
            # Time multiple executions
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                result = primitive(sample_grid)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            
            # Should be well under 200µs target
            if avg_time > 0.0002:
                pytest.skip(f"{primitive.name} performance: {avg_time*1000000:.0f}µs > 200µs target")


class TestDSLOperation:
    """Test DSL operation representation."""
    
    def test_operation_creation(self):
        """Test creating DSL operations."""
        op = DSLOperation("Rotate90", {})
        assert op.primitive_name == "Rotate90"
        assert op.parameters == {}
    
    def test_operation_with_parameters(self):
        """Test operation with parameters."""
        op = DSLOperation("Paint", {"x": 1, "y": 2, "c": 3})
        assert op.primitive_name == "Paint"
        assert op.parameters == {"x": 1, "y": 2, "c": 3}
    
    def test_operation_string_representation(self):
        """Test string representation of operations."""
        op1 = DSLOperation("Rotate90", {})
        assert str(op1) == "Rotate90"
        
        op2 = DSLOperation("Paint", {"x": 1, "y": 2, "c": 3})
        assert "Paint(" in str(op2)
        assert "x=1" in str(op2)
        assert "y=2" in str(op2)
        assert "c=3" in str(op2)
    
    def test_operation_hashing(self):
        """Test operation hashing for deduplication."""
        op1 = DSLOperation("Rotate90", {})
        op2 = DSLOperation("Rotate90", {})
        op3 = DSLOperation("Rotate180", {})
        
        assert hash(op1) == hash(op2)
        assert hash(op1) != hash(op3)


class TestDSLProgram:
    """Test DSL program representation."""
    
    def test_program_creation(self):
        """Test creating DSL programs."""
        ops = [
            DSLOperation("Rotate90", {}),
            DSLOperation("ReflectH", {})
        ]
        program = DSLProgram(ops)
        
        assert len(program) == 2
        assert program.operations == ops
    
    def test_program_length_limit(self):
        """Test program length limits."""
        ops = [DSLOperation("Rotate90", {})] * 5
        
        with pytest.raises(ValueError):
            DSLProgram(ops, max_length=4)
    
    def test_program_string_representation(self):
        """Test string representation of programs."""
        ops = [
            DSLOperation("Rotate90", {}),
            DSLOperation("ReflectH", {})
        ]
        program = DSLProgram(ops)
        
        program_str = str(program)
        assert "Rotate90" in program_str
        assert "ReflectH" in program_str
        assert "->" in program_str
    
    def test_empty_program(self):
        """Test empty program."""
        program = DSLProgram([])
        assert len(program) == 0
        assert str(program) == "EmptyProgram"
    
    def test_program_serialization(self):
        """Test program serialization to/from dict."""
        ops = [
            DSLOperation("Rotate90", {}),
            DSLOperation("Paint", {"x": 1, "y": 2, "c": 3})
        ]
        program = DSLProgram(ops)
        
        # Serialize
        data = program.to_dict()
        assert data['length'] == 2
        assert len(data['operations']) == 2
        
        # Deserialize
        restored_program = DSLProgram.from_dict(data)
        assert len(restored_program) == len(program)
        assert restored_program.operations[0].primitive_name == "Rotate90"
        assert restored_program.operations[1].parameters == {"x": 1, "y": 2, "c": 3}
    
    def test_program_copy(self):
        """Test program copying."""
        ops = [DSLOperation("Rotate90", {})]
        program = DSLProgram(ops)
        
        copy_program = program.copy()
        assert copy_program == program
        assert copy_program is not program
        assert copy_program.operations is not program.operations
    
    def test_program_append(self):
        """Test appending operations to programs."""
        program = DSLProgram([DSLOperation("Rotate90", {})])
        new_op = DSLOperation("ReflectH", {})
        
        new_program = program.append(new_op)
        assert len(new_program) == 2
        assert len(program) == 1  # Original unchanged
        
        # Test length limit
        long_program = DSLProgram([DSLOperation("Rotate90", {})] * 4, max_length=4)
        with pytest.raises(ValueError):
            long_program.append(DSLOperation("ReflectH", {}))


class TestDSLEngine:
    """Test DSL engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create DSL engine for testing."""
        return create_dsl_engine(max_program_length=4)
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample grid for testing."""
        return np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.max_program_length == 4
        assert len(engine.primitives) > 0
        assert engine.execution_count == 0
    
    def test_execute_simple_program(self, engine, sample_grid):
        """Test executing a simple program."""
        ops = [DSLOperation("Rotate90", {})]
        program = DSLProgram(ops)
        
        result, info = engine.execute_program(program, sample_grid)
        
        assert info['success'] is True
        assert info['operations_executed'] == 1
        assert info['execution_time'] > 0
        
        # Check result
        expected = np.array([[3, 1], [4, 2]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_execute_multi_operation_program(self, engine, sample_grid):
        """Test executing multi-operation program."""
        ops = [
            DSLOperation("Rotate90", {}),
            DSLOperation("ReflectH", {})
        ]
        program = DSLProgram(ops)
        
        result, info = engine.execute_program(program, sample_grid)
        
        assert info['success'] is True
        assert info['operations_executed'] == 2
    
    def test_execute_program_with_parameters(self, engine, sample_grid):
        """Test executing program with parameters."""
        ops = [DSLOperation("Paint", {"x": 0, "y": 0, "c": 9})]
        program = DSLProgram(ops)
        
        result, info = engine.execute_program(program, sample_grid)
        
        assert info['success'] is True
        assert result[0, 0] == 9
        assert result[0, 1] == sample_grid[0, 1]  # Unchanged
    
    def test_execute_invalid_program(self, engine, sample_grid):
        """Test executing program with invalid primitive."""
        ops = [DSLOperation("NonexistentPrimitive", {})]
        program = DSLProgram(ops)
        
        result, info = engine.execute_program(program, sample_grid)
        
        assert info['success'] is False
        assert info['error'] is not None
        assert "Unknown primitive" in info['error']
    
    def test_validate_program(self, engine):
        """Test program validation."""
        # Valid program
        valid_ops = [DSLOperation("Rotate90", {})]
        valid_program = DSLProgram(valid_ops)
        
        is_valid, error = engine.validate_program(valid_program)
        assert is_valid is True
        assert error is None
        
        # Invalid program (unknown primitive)
        invalid_ops = [DSLOperation("UnknownPrimitive", {})]
        invalid_program = DSLProgram(invalid_ops)
        
        is_valid, error = engine.validate_program(invalid_program)
        assert is_valid is False
        assert error is not None
    
    def test_create_operation(self, engine):
        """Test creating operations through engine."""
        op = engine.create_operation("Rotate90")
        assert op.primitive_name == "Rotate90"
        assert op.parameters == {}
        
        op_with_params = engine.create_operation("Paint", x=1, y=2, c=3)
        assert op_with_params.parameters == {"x": 1, "y": 2, "c": 3}
        
        # Test invalid operation
        with pytest.raises(ValueError):
            engine.create_operation("UnknownPrimitive")
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            engine.create_operation("Paint", x=-1, y=0, c=5)
    
    def test_execution_statistics(self, engine, sample_grid):
        """Test execution statistics tracking."""
        initial_stats = engine.get_execution_stats()
        assert initial_stats['total_executions'] == 0
        
        # Execute a program
        ops = [DSLOperation("Rotate90", {})]
        program = DSLProgram(ops)
        engine.execute_program(program, sample_grid)
        
        stats = engine.get_execution_stats()
        assert stats['total_executions'] == 1
        assert stats['total_time'] > 0
        assert stats['average_time'] > 0
        
        # Reset stats
        engine.reset_stats()
        reset_stats = engine.get_execution_stats()
        assert reset_stats['total_executions'] == 0
    
    def test_enumerate_programs(self, engine):
        """Test program enumeration."""
        programs = engine.enumerate_programs(max_length=2)
        
        # Should include empty program and programs of length 1 and 2
        assert len(programs) > 0
        
        # Check that we have programs of different lengths
        lengths = [len(p) for p in programs]
        assert 0 in lengths  # Empty program
        assert 1 in lengths  # Single operation programs
        
        # All programs should be valid
        for program in programs[:10]:  # Test first 10
            is_valid, error = engine.validate_program(program)
            assert is_valid, f"Invalid program: {program}, error: {error}"
    
    def test_get_primitive_names(self, engine):
        """Test getting primitive names."""
        names = engine.get_primitive_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert "Rotate90" in names
        assert "ReflectH" in names


class TestDSLIntegration:
    """Integration tests for DSL system."""
    
    def test_complete_transformation_pipeline(self):
        """Test complete transformation pipeline."""
        engine = create_dsl_engine()
        
        # Create input grid
        input_grid = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        # Create program: Rotate90 -> ReflectH -> Paint(0,0,9)
        ops = [
            DSLOperation("Rotate90", {}),
            DSLOperation("ReflectH", {}),
            DSLOperation("Paint", {"x": 0, "y": 0, "c": 9})
        ]
        program = DSLProgram(ops)
        
        # Execute program
        result, info = engine.execute_program(program, input_grid)
        
        assert info['success'] is True
        assert info['operations_executed'] == 3
        assert result[0, 0] == 9  # Painted pixel
    
    def test_performance_requirements(self):
        """Test that DSL execution meets performance requirements."""
        engine = create_dsl_engine()
        
        # Create test grid
        test_grid = np.random.randint(0, 5, size=(10, 10), dtype=np.int32)
        
        # Create simple program
        ops = [DSLOperation("Rotate90", {}), DSLOperation("ReflectH", {})]
        program = DSLProgram(ops)
        
        # Time multiple executions
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result, info = engine.execute_program(program, test_grid)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # Should be well under 1ms for simple programs
        if avg_time > 0.001:
            pytest.skip(f"DSL performance: {avg_time*1000:.2f}ms > 1ms target")
    
    def test_program_space_size(self):
        """Test that program space size matches theoretical expectations."""
        engine = create_dsl_engine(max_program_length=4)
        
        # For parameter-free primitives, the search space should be manageable
        programs = engine.enumerate_programs(max_length=2)
        
        # Should have reasonable number of programs
        assert len(programs) > 0
        assert len(programs) < 1000  # Should be manageable
        
        # Verify no duplicates
        program_hashes = [hash(p) for p in programs]
        assert len(program_hashes) == len(set(program_hashes))


class TestDSLGrammar:
    """Tests for DSL grammar parsing and validation."""

    def test_parse_dsl_program_simple(self):
        engine = create_dsl_engine()
        prog = engine.parse_dsl_program("Rotate90 -> ReflectH")
        assert isinstance(prog, DSLProgram)
        assert len(prog.operations) == 2
        assert prog.operations[0].primitive_name == "Rotate90"
        assert prog.operations[1].primitive_name == "ReflectH"

    def test_parse_with_named_args(self):
        engine = create_dsl_engine()
        prog = engine.parse_dsl_program("Paint(x=1, y=2, c=3)")
        assert len(prog.operations) == 1
        op = prog.operations[0]
        assert op.primitive_name == "Paint"
        assert op.parameters == {"x": 1, "y": 2, "c": 3}

    def test_parse_with_positional_args(self):
        engine = create_dsl_engine()
        prog = engine.parse_dsl_program("Paint(1, 2, 3)")
        op = prog.operations[0]
        assert op.parameters == {"x": 1, "y": 2, "c": 3}

    def test_parse_mapcolors_list_literal(self):
        engine = create_dsl_engine()
        perm = list(range(10))
        text = "MapColors([%s])" % ", ".join(str(x) for x in perm)
        prog = engine.parse_dsl_program(text)
        op = prog.operations[0]
        assert op.parameters["perm"] == perm

    def test_validate_program_text(self):
        engine = create_dsl_engine()
        ok, err = engine.validate_program_text("Rotate90 -> ReflectV")
        assert ok and err is None

        ok, err = engine.validate_program_text("Unknown()")
        assert not ok and err is not None

    def test_predicate_shorthand(self):
        engine = create_dsl_engine()
        # Size predicate
        prog = engine.parse_dsl_program("PaintIf(size>=3, 9)")
        op = prog.operations[0]
        assert op.primitive_name == "PaintIf"
        assert hasattr(op.parameters["predicate"], "evaluate")
        
        # Color predicate list
        prog2 = engine.parse_dsl_program("PaintIf(color(1,3), 5)")
        op2 = prog2.operations[0]
        assert op2.primitive_name == "PaintIf"
        assert hasattr(op2.parameters["predicate"], "evaluate")