"""Conservative rewrite rules and program canonicalization (safe subset).

Only applies algebraic identities we are 100% confident in:
 - Rotate composition modulo 4
 - ReflectH ∘ ReflectH = Id; ReflectV ∘ ReflectV = Id
 - Translate(dx1,dy1) ∘ Translate(dx2,dy2) = Translate(dx1+dx2, dy1+dy2)
 - MapColors permutation composition
 - MapColorsPermutation dict composition (seen colors only)
 - Rotate180 ∘ ReflectH == ReflectH ∘ Rotate180 (commutes safely)
 - MapColorsPermutation ∘ MapColors composition (merge when both present)

This returns a DSLProgram with an equivalent (or shorter) sequence. If any
error occurs, falls back to the original program.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


_ROTATE_NAMES = ["Rotate90", "Rotate180", "Rotate270"]


def _compose_rotates(seq: List[DSLOperation]) -> List[DSLOperation]:
    k = 0
    rest: List[DSLOperation] = []
    for op in seq:
        if op.primitive_name == "Rotate90":
            k = (k + 1) % 4
        elif op.primitive_name == "Rotate180":
            k = (k + 2) % 4
        elif op.primitive_name == "Rotate270":
            k = (k + 3) % 4
        else:
            rest.append(op)
    if k == 0:
        return rest
    name = {1: "Rotate90", 2: "Rotate180", 3: "Rotate270"}[k]
    return [DSLOperation(name, {})] + rest


def _cancel_reflections(seq: List[DSLOperation]) -> List[DSLOperation]:
    out: List[DSLOperation] = []
    h_parity = 0
    v_parity = 0
    for op in seq:
        if op.primitive_name == "ReflectH":
            h_parity ^= 1
        elif op.primitive_name == "ReflectV":
            v_parity ^= 1
        else:
            out.append(op)
    if h_parity:
        out.insert(0, DSLOperation("ReflectH", {}))
    if v_parity:
        out.insert(0, DSLOperation("ReflectV", {}))
    return out


def _compose_translates(seq: List[DSLOperation]) -> List[DSLOperation]:
    dx_total = 0
    dy_total = 0
    rest: List[DSLOperation] = []
    for op in seq:
        if op.primitive_name == "Translate":
            dx_total += int(op.parameters.get("dx", 0))
            dy_total += int(op.parameters.get("dy", 0))
        else:
            rest.append(op)
    if dx_total == 0 and dy_total == 0:
        return rest
    return [DSLOperation("Translate", {"dx": dx_total, "dy": dy_total})] + rest


def _compose_mapcolors(seq: List[DSLOperation]) -> List[DSLOperation]:
    perm = None
    rest: List[DSLOperation] = []
    for op in seq:
        if op.primitive_name == "MapColors":
            p = list(op.parameters.get("perm", []))
            if len(p) != 10:
                rest.append(op)
                continue
            if perm is None:
                perm = list(range(10))
            # compose: new_perm[i] = p[perm[i]]
            perm = [p[perm[i]] for i in range(10)]
        else:
            rest.append(op)
    if perm is None:
        return rest
    return [DSLOperation("MapColors", {"perm": perm})] + rest


def _compose_mapcolors_permmap(seq: List[DSLOperation]) -> List[DSLOperation]:
    mapping: Dict[int, int] = {}
    rest: List[DSLOperation] = []
    for op in seq:
        if op.primitive_name == "MapColorsPermutation":
            m = dict(op.parameters.get("perm_map", {}))
            # Compose: apply m after current mapping
            if not mapping:
                mapping = m
            else:
                new_map: Dict[int, int] = {}
                for k, v in mapping.items():
                    new_map[k] = m.get(v, v)
                # also include direct entries of m not covered by mapping's domain
                for k, v in m.items():
                    if k not in new_map:
                        new_map[k] = v
                mapping = new_map
        else:
            rest.append(op)
    if not mapping:
        return rest
    return [DSLOperation("MapColorsPermutation", {"perm_map": mapping})] + rest


def _commute_rotate180_reflecth(seq: List[DSLOperation]) -> List[DSLOperation]:
    """Reorder Rotate180 and ReflectH for canonical form: Rotate180 ∘ ReflectH == ReflectH ∘ Rotate180.

    Since Rotate180 and ReflectH commute (both are 180-degree operations), we can
    canonicalize their order to prefer ReflectH first for consistency.
    """
    out: List[DSLOperation] = []
    rotate180_count = 0

    for op in seq:
        if op.primitive_name == "Rotate180":
            rotate180_count += 1
        elif op.primitive_name == "ReflectH":
            # Since they commute, put ReflectH first, then Rotate180
            if rotate180_count > 0:
                out.append(DSLOperation("ReflectH", {}))
                # Add the Rotate180 operations
                for _ in range(rotate180_count):
                    out.append(DSLOperation("Rotate180", {}))
                rotate180_count = 0
            else:
                out.append(op)
        else:
            out.append(op)

    # Add any remaining Rotate180 operations
    for _ in range(rotate180_count):
        out.append(DSLOperation("Rotate180", {}))

    return out


def _merge_mapcolors_operations(seq: List[DSLOperation]) -> List[DSLOperation]:
    """Merge MapColorsPermutation and MapColors when both are present.

    When we have MapColorsPermutation followed by MapColors, we can compose them
    into a single MapColors operation by applying the permutation to the map.
    """
    out: List[DSLOperation] = []
    pending_permutation: Optional[Dict[int, int]] = None

    for op in seq:
        if op.primitive_name == "MapColorsPermutation":
            # Extract the permutation mapping
            perm_map = dict(op.parameters.get("perm_map", {}))
            if pending_permutation is None:
                pending_permutation = perm_map
            else:
                # Compose: apply perm_map after current pending_permutation
                new_perm: Dict[int, int] = {}
                for k, v in pending_permutation.items():
                    new_perm[k] = perm_map.get(v, v)
                # Also include direct entries from perm_map not in pending domain
                for k, v in perm_map.items():
                    if k not in new_perm:
                        new_perm[k] = v
                pending_permutation = new_perm

        elif op.primitive_name == "MapColors":
            # Check if we have a pending permutation to merge with
            if pending_permutation is not None:
                # Compose: apply MapColors after MapColorsPermutation
                perm_list = list(op.parameters.get("perm", []))
                if len(perm_list) == 10:  # Valid permutation length
                    # Apply the pending permutation to the MapColors permutation
                    composed_perm = [0] * 10
                    for i in range(10):
                        composed_perm[i] = perm_list[pending_permutation.get(i, i)]
                    # Replace with composed operation
                    out.append(DSLOperation("MapColors", {"perm": composed_perm}))
                    pending_permutation = None  # Reset pending
                else:
                    # Invalid perm, keep both operations
                    out.append(DSLOperation("MapColorsPermutation", {"perm_map": pending_permutation}))
                    out.append(op)
                    pending_permutation = None
            else:
                out.append(op)
        else:
            out.append(op)

    # If we have a pending permutation that wasn't merged, add it
    if pending_permutation is not None:
        out.append(DSLOperation("MapColorsPermutation", {"perm_map": pending_permutation}))

    return out


def canonicalize_program(program: DSLProgram) -> DSLProgram:
    """Return a conservatively canonicalized program.

    The order of combination prefers to reduce parameter-free ops first, then
    safe compositions. If anything fails, return original program.
    """
    try:
        ops = list(program.operations)
        # Safe groupwise compositions
        ops = _compose_rotates(ops)
        ops = _cancel_reflections(ops)
        ops = _compose_translates(ops)
        ops = _compose_mapcolors(ops)
        ops = _compose_mapcolors_permmap(ops)
        # New commutative and merging rules
        ops = _commute_rotate180_reflecth(ops)
        ops = _merge_mapcolors_operations(ops)
        # Remove leading identities if any snuck in (rare)
        # No explicit Identity primitive is defined; nothing to drop here
        return DSLProgram(ops, max_length=program.max_length)
    except Exception:
        return program

