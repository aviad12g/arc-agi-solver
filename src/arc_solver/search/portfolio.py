"""Portfolio scheduler with early-stop orchestration.

Runs multiple solving lanes with individual time budgets and stops immediately
when any lane proves a solution over all training examples. Designed to be
called from the CLI layer and reuse existing components.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram
from arc_solver.search.llm_integration import LLMGuidedAStarSearcher


_PORTFOLIO_STATS: Dict[str, Dict[str, float]] = {
    # lane -> {'calls': int, 'wins': int, 'time': float}
}


def _record_lane(lane: str, win: bool, duration: float) -> None:
    s = _PORTFOLIO_STATS.setdefault(lane, {'calls': 0.0, 'wins': 0.0, 'time': 0.0})
    s['calls'] += 1.0
    if win:
        s['wins'] += 1.0
    s['time'] += max(0.0, float(duration))


def get_portfolio_stats() -> Dict[str, Dict[str, float]]:
    return {k: dict(v) for k, v in _PORTFOLIO_STATS.items()}


class PortfolioRunner:
    def __init__(
        self,
        dsl_engine: DSLEngine,
        searcher,  # AStarSearcher
        llm_proposer,  # Optional[LLMProposer]
        retrieval_index,  # RetrievalIndex
        logger,
    ) -> None:
        self.dsl_engine = dsl_engine
        self.searcher = searcher
        self.llm_proposer = llm_proposer
        self.retrieval_index = retrieval_index
        self.logger = logger

    def _validate_program_on_train(self, program: DSLProgram, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for a, b in train_pairs:
            try:
                pred, _ = self.dsl_engine.execute_program(program, a)
            except Exception:
                return False
            if not np.array_equal(pred, b):
                return False
        return True

    def _result_from_program(self, program: DSLProgram, test_inputs: List[np.ndarray], task_id: Optional[str] = None,
                             portfolio_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        test_predictions = []
        for ti in test_inputs:
            pred, _ = self.dsl_engine.execute_program(program, ti)
            test_predictions.append(pred.tolist())
        res = {
            'success': True,
            'program': program.to_dict(),
            'predictions': test_predictions,
            'search_stats': {
                'nodes_expanded': 0,
                'nodes_generated': 0,
                'max_depth_reached': len(program.operations),
                'termination_reason': 'portfolio_solution'
            },
            'task_id': task_id,
        }
        if portfolio_metrics:
            res['search_stats']['portfolio_metrics'] = portfolio_metrics
        return res

    def run(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        task_id: Optional[str],
        budgets: Dict[str, float],  # seconds per lane
        use_multi_example: bool,
        llm_priority_boost: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        metrics: Dict[str, float] = {}

        # Hazard-aware reordering of lanes based on historical efficiency (wins / time)
        try:
            from arc_solver.config import get_config as _get_cfg
            cfg = _get_cfg()
            hazard_sched = False
            if cfg is not None and 'search' in cfg and 'portfolio' in cfg.search:
                hazard_sched = bool(cfg.search.portfolio.get('hazard_scheduling', False))
        except Exception:
            hazard_sched = False
        lane_order = ['retrieval', 'templates', 'object_level', 'cegis', 'astar', 'puct', 'llm_guided']
        if hazard_sched and _PORTFOLIO_STATS:
            def efficiency(lane: str) -> float:
                s = _PORTFOLIO_STATS.get(lane, None)
                if not s or s['time'] <= 0.0:
                    return 0.0
                return s['wins'] / max(1e-6, s['time'])
            lane_order.sort(key=lambda l: -efficiency(l))

        # Micro-oracle pre-checks with UNSAT certificate reuse
        try:
            from arc_solver.search.micro_oracles import run_oracles
            from arc_solver.search.unsat_cache import UNSATCache, make_signature
            # Read UNSAT cache flag
            use_unsat = False
            signatures_file = None
            try:
                from arc_solver.config import get_config as _get_cfg
                cfg = _get_cfg()
                if cfg is not None and 'search' in cfg and 'advanced' in cfg.search and 'unsat_cache' in cfg.search.advanced:
                    ucfg = cfg.search.advanced.unsat_cache
                    use_unsat = bool(ucfg.get('enabled', False))
                    signatures_file = ucfg.get('signatures_file', None)
            except Exception:
                pass
            # If any train pair yields a program or UNSAT, handle accordingly
            lane_start = time.perf_counter()
            for a, b in train_pairs:
                o = run_oracles(a, b)
                # Program found by oracle → return immediately
                if o.program is not None:
                    metrics['winner'] = 'oracle'
                    metrics['oracle_time'] = time.perf_counter() - lane_start
                    return self._result_from_program(o.program, test_inputs, task_id, metrics)
                # UNSAT proof → record and short-circuit lanes
                if o.unsat and use_unsat:
                    sig = make_signature(a, b)
                    # Try to attach to searcher's cache if present
                    try:
                        if hasattr(self.searcher, '_unsat_cache') and self.searcher._unsat_cache is not None:
                            self.searcher._unsat_cache.add_unsat(sig)
                        else:
                            # Create ephemeral cache
                            _tmp_cache = UNSATCache()
                            _tmp_cache.add_unsat(sig)
                    except Exception:
                        pass
                    # Optionally append to signatures file for reuse across runs
                    try:
                        if signatures_file:
                            import json
                            with open(str(signatures_file), 'a') as f:
                                json.dump({'signature': sig, 'reason': 'oracle_unsat'}, f)
                                f.write('\n')
                    except Exception:
                        pass
                    # Stop portfolio lanes early; return None to allow outer code to continue fallback paths
                    self.logger.info("Portfolio: micro-oracle UNSAT detected; skipping lanes")
                    return None
        except Exception:
            pass

        # Lane: Retrieval
        if 'retrieval' in lane_order and budgets.get('retrieval', 0) > 0:
            self.logger.info("Portfolio: retrieval lane")
            lane_start = time.perf_counter()
            sig = None
            try:
                from arc_solver.search.retrieval import task_signature
                sig = task_signature(train_pairs)
                candidates = self.retrieval_index.get(sig)
                if not candidates:
                    try:
                        neighbors = self.retrieval_index.nearest(sig, k=3)
                        for nsig, _dist in neighbors:
                            cands = self.retrieval_index.get_by_signature(nsig, max_items=3)
                            candidates.extend(cands)
                    except Exception:
                        pass
            except Exception:
                candidates = []
            for prog in candidates:
                if self._validate_program_on_train(prog, train_pairs):
                    metrics['retrieval_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'retrieval'
                    dur = time.perf_counter() - lane_start
                    _record_lane('retrieval', True, dur)
                    return self._result_from_program(prog, test_inputs, task_id, metrics)
            metrics['retrieval_time'] = time.perf_counter() - lane_start
            if metrics['retrieval_time'] > budgets['retrieval']:
                _record_lane('retrieval', False, metrics['retrieval_time'])
                return None

        # Lane: Formula templates
        if 'templates' in lane_order and budgets.get('templates', 0) > 0:
            self.logger.info("Portfolio: formula templates lane")
            lane_start = time.perf_counter()
            try:
                from arc_solver.solver.formula_layer.solver import solve_with_templates
                prog = solve_with_templates(train_pairs, self.dsl_engine)
                if prog is not None and self._validate_program_on_train(prog, train_pairs):
                    metrics['templates_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'templates'
                    dur = time.perf_counter() - lane_start
                    _record_lane('templates', True, dur)
                    return self._result_from_program(prog, test_inputs, task_id, metrics)
            except Exception:
                pass
            metrics['templates_time'] = time.perf_counter() - lane_start
            if metrics['templates_time'] > budgets['templates']:
                _record_lane('templates', False, metrics['templates_time'])
                return None

        # Lane: Object-level synthesis
        if 'object_level' in lane_order and budgets.get('object_level', 0) > 0:
            self.logger.info("Portfolio: object-level synthesis lane")
            lane_start = time.perf_counter()
            try:
                from arc_solver.reasoning.object_synthesis import synthesize_object_level_program
                prog = synthesize_object_level_program(train_pairs, self.dsl_engine)
                if prog is not None and self._validate_program_on_train(prog, train_pairs):
                    metrics['object_level_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'object_level'
                    dur = time.perf_counter() - lane_start
                    _record_lane('object_level', True, dur)
                    return self._result_from_program(prog, test_inputs, task_id, metrics)
            except Exception:
                pass
            metrics['object_level_time'] = time.perf_counter() - lane_start
            if metrics['object_level_time'] > budgets['object_level']:
                _record_lane('object_level', False, metrics['object_level_time'])
                return None

        # Lane: CEGIS
        if 'cegis' in lane_order and budgets.get('cegis', 0) > 0:
            self.logger.info("Portfolio: CEGIS lane")
            lane_start = time.perf_counter()
            try:
                from arc_solver.reasoning.smt_cegis import try_cegis_solve
                prog = try_cegis_solve(train_pairs, max_length=4, dsl_engine=self.dsl_engine)
                if prog is not None and self._validate_program_on_train(prog, train_pairs):
                    metrics['cegis_time'] = time.perf_counter() - lane_start
                    dur = time.perf_counter() - lane_start
                    _record_lane('cegis', True, dur)
                    return self._result_from_program(prog, test_inputs, task_id, metrics)
            except Exception:
                pass
            metrics['cegis_time'] = time.perf_counter() - lane_start
            if metrics['cegis_time'] > budgets['cegis']:
                _record_lane('cegis', False, metrics['cegis_time'])
                return None

        # Lane: A*
        if 'astar' in lane_order and budgets.get('astar', 0) > 0:
            self.logger.info("Portfolio: A* lane")
            lane_start = time.perf_counter()
            try:
                # Enforce per-lane time budget by temporarily adjusting searcher config
                original_timeout = getattr(self.searcher.config, 'max_computation_time', 30.0)
                self.searcher.config.max_computation_time = float(budgets['astar'])
                if use_multi_example and len(train_pairs) > 1:
                    res = self.searcher.search_multi_example(train_pairs)
                else:
                    a, b = train_pairs[0]
                    res = self.searcher.search(a, b)
                # Restore timeout
                self.searcher.config.max_computation_time = original_timeout
                if res and res.success and res.program is not None:
                    metrics['astar_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'astar'
                    dur = time.perf_counter() - lane_start
                    _record_lane('astar', True, dur)
                    return self._result_from_program(res.program, test_inputs, task_id, metrics)
            except Exception:
                pass
            _record_lane('astar', False, time.perf_counter() - lane_start)
            # No strict time enforcement here; rely on searcher config

        # Lane: PUCT (verification-gated)
        if 'puct' in lane_order and budgets.get('puct', 0) > 0:
            self.logger.info("Portfolio: PUCT lane")
            lane_start = time.perf_counter()
            try:
                a, b = train_pairs[0]
                from arc_solver.search.puct import PUCTSearcher
                # Build op prior boosts from retrieval and LLM proposals when available
                op_boost: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], float] = {}
                try:
                    # Retrieval programs → boosts
                    from arc_solver.search.retrieval import task_signature
                    sig = task_signature(train_pairs)
                    r_progs = self.retrieval_index.get(sig, max_items=5)
                    # Also include nearest neighbor retrieved programs
                    if not r_progs:
                        try:
                            neighbors = self.retrieval_index.nearest(sig, k=3)
                            for nsig, _dist in neighbors:
                                r_progs.extend(self.retrieval_index.get_by_signature(nsig, max_items=3))
                        except Exception:
                            pass
                    for prog in r_progs[:10]:
                        for op in prog.operations:
                            key = (op.primitive_name, tuple(sorted(op.parameters.items())))
                            op_boost[key] = op_boost.get(key, 0.0) + 0.5
                except Exception:
                    pass
                try:
                    # LLM proposals → boosts
                    if self.llm_proposer is not None:
                        from arc_solver.perception.blob_labeling import create_blob_labeler
                        bl = create_blob_labeler(use_gpu=False)
                        in_blobs, _ = bl.label_blobs(a)
                        out_blobs, _ = bl.label_blobs(b)
                        prop = self.llm_proposer.generate_proposals(a, b, in_blobs, out_blobs)
                        llm_progs = prop.proposals if (prop and prop.success) else []
                        for prog in llm_progs:
                            for op in prog.operations:
                                key = (op.primitive_name, tuple(sorted(op.parameters.items())))
                                op_boost[key] = op_boost.get(key, 0.0) + 1.0
                except Exception:
                    pass

                puct = PUCTSearcher(self.dsl_engine, self.searcher.heuristic_system, c_puct=1.0, max_depth=self.searcher.config.max_program_length, time_budget=budgets['puct'], op_prior_boost=op_boost)
                prog = puct.search(a, b, train_pairs if (use_multi_example and len(train_pairs) > 1) else [(a, b)])
                if prog is not None:
                    metrics['puct_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'puct'
                    dur = time.perf_counter() - lane_start
                    _record_lane('puct', True, dur)
                    return self._result_from_program(prog, test_inputs, task_id, metrics)
            except Exception:
                pass
            _record_lane('puct', False, time.perf_counter() - lane_start)

        # Lane: LLM-guided A*
        if 'llm_guided' in lane_order and budgets.get('llm_guided', 0) > 0 and self.llm_proposer is not None:
            self.logger.info("Portfolio: LLM-guided A* lane")
            lane_start = time.perf_counter()
            try:
                a, b = train_pairs[0]
                # Prepare blobs for proposer
                from arc_solver.perception.blob_labeling import create_blob_labeler
                bl = create_blob_labeler(use_gpu=False)
                in_blobs, _ = bl.label_blobs(a)
                out_blobs, _ = bl.label_blobs(b)
                prop = self.llm_proposer.generate_proposals(a, b, in_blobs, out_blobs)
                llm_programs = prop.proposals if (prop and prop.success) else []
                guided = LLMGuidedAStarSearcher(self.searcher, llm_programs, llm_priority_boost, self.dsl_engine)
                original_timeout = getattr(self.searcher.config, 'max_computation_time', 30.0)
                self.searcher.config.max_computation_time = float(budgets['llm_guided'])
                res = guided.search(a, b, train_pairs if (use_multi_example and len(train_pairs) > 1) else None)
                self.searcher.config.max_computation_time = original_timeout
                if res and res.success and res.program is not None:
                    metrics['llm_guided_time'] = time.perf_counter() - lane_start
                    metrics['winner'] = 'llm_guided'
                    dur = time.perf_counter() - lane_start
                    _record_lane('llm_guided', True, dur)
                    return self._result_from_program(res.program, test_inputs, task_id, metrics)
            except Exception:
                pass
            _record_lane('llm_guided', False, time.perf_counter() - lane_start)

        # Persist metrics as JSONL if configured
        try:
            from arc_solver.config import get_config as _get_cfg
            cfg = _get_cfg()
            record = False
            metrics_file = None
            if cfg is not None and 'search' in cfg and 'portfolio' in cfg.search:
                pcfg = cfg.search.portfolio
                record = bool(pcfg.get('record_metrics', False))
                metrics_file = pcfg.get('metrics_file', None)
            if record and metrics_file and metrics:
                import json
                rec = dict(metrics)
                rec['task_id'] = task_id
                rec['budgets'] = budgets
                with open(str(metrics_file), 'a') as f:
                    json.dump(rec, f)
                    f.write('\n')
        except Exception:
            pass

        return None
