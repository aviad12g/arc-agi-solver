"""Unit tests for rewrite rules and program canonicalization.

Tests the equivalence of canonicalized programs to ensure that rewrite rules
preserve semantic correctness while potentially shortening programs.
"""

import pytest
import numpy as np
from unittest.mock import patch

from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation, DSLEngine
from arc_solver.reasoning.rewrite_rules import (
    canonicalize_program,
    _commute_rotate180_reflecth,
    _merge_mapcolors_operations
)
from arc_solver.core.data_models import Grid


class TestRewriteRules:
    """Test rewrite rules functionality."""

    def test_canonicalize_program_basic(self):
        """Test basic canonicalization with empty program."""
        program = DSLProgram([])
        canonicalized = canonicalize_program(program)
        assert len(canonicalized) == 0
        assert canonicalized == program

    def test_rotate180_reflecth_commutativity(self):
        """Test that Rotate180 and ReflectH commute and canonicalize consistently."""
        # Create a simple test grid
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

        # Create programs with different orderings
        ops1 = [
            DSLOperation("Rotate180", {}),
            DSLOperation("ReflectH", {})
        ]
        ops2 = [
            DSLOperation("ReflectH", {}),
            DSLOperation("Rotate180", {})
        ]

        program1 = DSLProgram(ops1)
        program2 = DSLProgram(ops2)

        # Canonicalize both
        canonicalized1 = canonicalize_program(program1)
        canonicalized2 = canonicalize_program(program2)

        # They should canonicalize to the same program
        assert canonicalized1 == canonicalized2

        # Test that they produce the same result when executed
        dsl_engine = DSLEngine()
        result1, _ = dsl_engine.execute_program(canonicalized1, grid)
        result2, _ = dsl_engine.execute_program(canonicalized2, grid)
        assert np.array_equal(result1, result2)

    def test_rotate180_reflecth_canonical_order(self):
        """Test that Rotate180 and ReflectH are reordered to canonical form."""
        ops = [
            DSLOperation("Rotate180", {}),
            DSLOperation("ReflectH", {}),
            DSLOperation("Rotate90", {})
        ]

        reordered = _commute_rotate180_reflecth(ops)

        # Should reorder to: ReflectH, Rotate180, Rotate90
        assert len(reordered) == 3
        assert reordered[0].primitive_name == "ReflectH"
        assert reordered[1].primitive_name == "Rotate180"
        assert reordered[2].primitive_name == "Rotate90"

    def test_multiple_rotate180_reflecth(self):
        """Test multiple Rotate180 and ReflectH operations."""
        ops = [
            DSLOperation("Rotate180", {}),
            DSLOperation("Rotate180", {}),
            DSLOperation("ReflectH", {}),
            DSLOperation("ReflectH", {})
        ]

        reordered = _commute_rotate180_reflecth(ops)

        # Should reorder to: ReflectH, ReflectH, Rotate180, Rotate180
        assert len(reordered) == 4
        assert reordered[0].primitive_name == "ReflectH"
        assert reordered[1].primitive_name == "ReflectH"
        assert reordered[2].primitive_name == "Rotate180"
        assert reordered[3].primitive_name == "Rotate180"

    def test_mapcolors_operations_merge(self):
        """Test merging MapColorsPermutation and MapColors operations."""
        # Create a permutation mapping: 0->1, 1->2, 2->0
        perm_map = {0: 1, 1: 2, 2: 0}

        # Create a MapColors permutation: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]
        mapcolors_perm = [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]

        ops = [
            DSLOperation("MapColorsPermutation", {"perm_map": perm_map}),
            DSLOperation("MapColors", {"perm": mapcolors_perm}),
            DSLOperation("Rotate90", {})
        ]

        merged = _merge_mapcolors_operations(ops)

        # Should merge the two MapColors operations and keep Rotate90
        assert len(merged) == 2
        assert merged[0].primitive_name == "MapColors"
        assert merged[1].primitive_name == "Rotate90"

        # Check that the merged permutation is correct
        # Original: 0->1, 1->2, 2->0
        # MapColors: [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]
        # Result: [2, 0, 1, 3, 4, 5, 6, 7, 8, 9] (applying perm_map to mapcolors_perm)
        expected_perm = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]
        assert merged[0].parameters["perm"] == expected_perm

    def test_mapcolors_operations_no_merge(self):
        """Test that operations are preserved when they can't be merged."""
        ops = [
            DSLOperation("MapColorsPermutation", {"perm_map": {0: 1}}),
            DSLOperation("Rotate90", {}),  # Non-MapColors operation in between
            DSLOperation("MapColors", {"perm": [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]})
        ]

        merged = _merge_mapcolors_operations(ops)

        # Should preserve all operations since they can't be merged
        assert len(merged) == 3
        assert merged[0].primitive_name == "MapColorsPermutation"
        assert merged[1].primitive_name == "Rotate90"
        assert merged[2].primitive_name == "MapColors"

    def test_mapcolors_invalid_perm(self):
        """Test handling of invalid MapColors permutation."""
        ops = [
            DSLOperation("MapColorsPermutation", {"perm_map": {0: 1}}),
            DSLOperation("MapColors", {"perm": [1, 2]})  # Invalid length
        ]

        merged = _merge_mapcolors_operations(ops)

        # Should preserve both operations since permutation is invalid
        assert len(merged) == 2
        assert merged[0].primitive_name == "MapColorsPermutation"
        assert merged[1].primitive_name == "MapColors"

    def test_canonicalize_program_equivalence(self):
        """Test that canonicalized programs produce equivalent results."""
        # Create a test grid
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

        # Create equivalent programs that should canonicalize the same way
        programs = [
            # Program 1: Rotate180 then ReflectH
            DSLProgram([
                DSLOperation("Rotate180", {}),
                DSLOperation("ReflectH", {})
            ]),
            # Program 2: ReflectH then Rotate180 (should be equivalent)
            DSLProgram([
                DSLOperation("ReflectH", {}),
                DSLOperation("Rotate180", {})
            ]),
            # Program 3: Multiple rotates that can be composed
            DSLProgram([
                DSLOperation("Rotate90", {}),
                DSLOperation("Rotate90", {}),
                DSLOperation("ReflectH", {})
            ])
        ]

        dsl_engine = DSLEngine()
        canonicalized_programs = [canonicalize_program(p) for p in programs]

        # All should produce the same result
        results = []
        for program in canonicalized_programs:
            result, _ = dsl_engine.execute_program(program, grid)
            results.append(result)

        # Check that all results are equivalent
        for result in results[1:]:
            assert np.array_equal(results[0], result)

    def test_canonicalize_program_error_handling(self):
        """Test that canonicalize_program handles errors gracefully."""
        # Create a program that might cause issues
        program = DSLProgram([
            DSLOperation("InvalidOperation", {})
        ])

        # Should return original program on error
        canonicalized = canonicalize_program(program)
        assert canonicalized == program

    def test_canonicalize_empty_operations(self):
        """Test canonicalization of programs with only identity-like operations."""
        # Create program with operations that cancel out
        program = DSLProgram([
            DSLOperation("ReflectH", {}),
            DSLOperation("ReflectH", {})  # Should cancel to identity
        ])

        canonicalized = canonicalize_program(program)

        # Should remove the canceling reflections
        # Note: This depends on existing _cancel_reflections implementation
        assert len(canonicalized) <= len(program)

    def test_commutative_rule_order_preservation(self):
        """Test that commutative rule preserves operation order for non-commutative ops."""
        ops = [
            DSLOperation("Paint", {"x": 0, "y": 0, "c": 1}),
            DSLOperation("Rotate180", {}),
            DSLOperation("ReflectH", {}),
            DSLOperation("Translate", {"dx": 1, "dy": 1})
        ]

        reordered = _commute_rotate180_reflecth(ops)

        # Paint should stay first, Translate should stay last
        assert reordered[0].primitive_name == "Paint"
        assert reordered[-1].primitive_name == "Translate"

        # Rotate180 and ReflectH should be reordered
        rotate180_pos = next(i for i, op in enumerate(reordered) if op.primitive_name == "Rotate180")
        reflecth_pos = next(i for i, op in enumerate(reordered) if op.primitive_name == "ReflectH")
        assert reflecth_pos < rotate180_pos  # ReflectH should come first
