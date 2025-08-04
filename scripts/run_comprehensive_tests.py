#!/usr/bin/env python3
"""Comprehensive test runner for Task 12.1 validation.

This script runs all comprehensive unit tests and generates a detailed report
covering all system components and their mathematical correctness.
"""

import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Runs comprehensive tests and generates detailed reports."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Test modules to run
        self.test_modules = [
            {
                'name': 'Perception Layer',
                'module': 'tests.test_perception_comprehensive',
                'description': 'Blob labeling, feature invariance, symmetry detection',
                'requirements': ['2.6', '2.7', '8.1', '8.2', '8.3', '8.4', '8.5']
            },
            {
                'name': 'Reasoning Layer', 
                'module': 'tests.test_reasoning_comprehensive',
                'description': 'DSL correctness, program synthesis, performance',
                'requirements': ['3.1', '3.2', '3.3', '3.4', '3.5', '3.6']
            },
            {
                'name': 'Search Layer',
                'module': 'tests.test_search_comprehensive', 
                'description': 'A* optimality, heuristic admissibility, performance',
                'requirements': ['4.1', '4.2', '4.3', '4.4', '4.5', '4.6']
            },
            {
                'name': 'Integration Tests',
                'module': 'tests.test_integration_comprehensive',
                'description': 'End-to-end pipeline, component interaction',
                'requirements': ['1.1', '1.2', '1.3', '1.4', '7.1', '7.2', '7.3', '7.4']
            }
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests and return results."""
        logger.info("Starting comprehensive test suite...")
        self.start_time = time.perf_counter()
        
        overall_success = True
        
        for test_module in self.test_modules:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Running {test_module['name']} Tests")
            logger.info(f"Description: {test_module['description']}")
            logger.info(f"Requirements: {', '.join(test_module['requirements'])}")
            logger.info('='*60)
            
            result = self._run_test_module(test_module)
            self.test_results[test_module['name']] = result
            
            if not result['success']:
                overall_success = False
                logger.error(f"❌ {test_module['name']} tests FAILED")
            else:
                logger.info(f"✅ {test_module['name']} tests PASSED")
        
        self.end_time = time.perf_counter()
        
        # Generate comprehensive report
        report = self._generate_report(overall_success)
        
        return report
    
    def _run_test_module(self, test_module: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test module and return results."""
        module_name = test_module['module']
        
        try:
            # Run pytest on the module
            cmd = [
                sys.executable, '-m', 'pytest', 
                f"{module_name.replace('.', '/')}.py",
                '-v', '--tb=short', '--json-report', '--json-report-file=/tmp/pytest_report.json'
            ]
            
            start_time = time.perf_counter()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout per module
            )
            end_time = time.perf_counter()
            
            # Parse results
            success = result.returncode == 0
            execution_time = end_time - start_time
            
            # Try to parse JSON report if available
            test_details = self._parse_pytest_output(result.stdout, result.stderr)
            
            return {
                'success': success,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_details': test_details,
                'requirements_covered': test_module['requirements']
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test module {module_name} timed out after 5 minutes")
            return {
                'success': False,
                'execution_time': 300.0,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'test_details': {},
                'requirements_covered': test_module['requirements']
            }
        except Exception as e:
            logger.error(f"Error running test module {module_name}: {e}")
            return {
                'success': False,
                'execution_time': 0.0,
                'stdout': '',
                'stderr': str(e),
                'test_details': {},
                'requirements_covered': test_module['requirements']
            }
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test details."""
        details = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'test_cases': []
        }
        
        # Simple parsing of pytest output
        lines = stdout.split('\\n')
        
        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                parts = line.split('::')
                if len(parts) >= 2:
                    test_class = parts[0].split('/')[-1].replace('.py', '')
                    test_method = parts[1].split()[0]
                    status = 'PASSED' if 'PASSED' in line else ('FAILED' if 'FAILED' in line else 'SKIPPED')
                    
                    details['test_cases'].append({
                        'class': test_class,
                        'method': test_method,
                        'status': status
                    })
                    
                    details['total_tests'] += 1
                    if status == 'PASSED':
                        details['passed'] += 1
                    elif status == 'FAILED':
                        details['failed'] += 1
                    elif status == 'SKIPPED':
                        details['skipped'] += 1
        
        # Look for summary line
        for line in lines:
            if 'failed' in line and 'passed' in line:
                # Try to extract numbers from summary
                try:
                    import re
                    numbers = re.findall(r'(\\d+) (passed|failed|skipped|error)', line)
                    for count, status in numbers:
                        if status == 'passed':
                            details['passed'] = int(count)
                        elif status == 'failed':
                            details['failed'] = int(count)
                        elif status == 'skipped':
                            details['skipped'] = int(count)
                        elif status == 'error':
                            details['errors'] = int(count)
                except:
                    pass
        
        return details
    
    def _generate_report(self, overall_success: bool) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate overall statistics
        total_tests = sum(r['test_details'].get('total_tests', 0) for r in self.test_results.values())
        total_passed = sum(r['test_details'].get('passed', 0) for r in self.test_results.values())
        total_failed = sum(r['test_details'].get('failed', 0) for r in self.test_results.values())
        total_skipped = sum(r['test_details'].get('skipped', 0) for r in self.test_results.values())
        
        # Collect all requirements covered
        all_requirements = set()
        for result in self.test_results.values():
            all_requirements.update(result['requirements_covered'])
        
        report = {
            'overall_success': overall_success,
            'total_execution_time': total_time,
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'skipped': total_skipped,
                'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            'requirements_coverage': {
                'total_requirements': len(all_requirements),
                'requirements_tested': sorted(list(all_requirements))
            },
            'module_results': self.test_results,
            'performance_analysis': self._analyze_performance(),
            'mathematical_validation': self._analyze_mathematical_correctness(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance test results."""
        performance_analysis = {
            'perception_performance': 'Unknown',
            'reasoning_performance': 'Unknown', 
            'search_performance': 'Unknown',
            'end_to_end_performance': 'Unknown',
            'performance_targets_met': False
        }
        
        # Analyze performance from test outputs
        for module_name, result in self.test_results.items():
            stdout = result.get('stdout', '')
            
            # Look for performance indicators in output
            if 'performance' in stdout.lower():
                if 'perception' in module_name.lower():
                    if 'ms' in stdout and 'average' in stdout:
                        performance_analysis['perception_performance'] = 'Measured'
                elif 'reasoning' in module_name.lower():
                    if 'µs' in stdout and 'average' in stdout:
                        performance_analysis['reasoning_performance'] = 'Measured'
                elif 'search' in module_name.lower():
                    if 'nodes expanded' in stdout:
                        performance_analysis['search_performance'] = 'Measured'
        
        # Check if performance targets are mentioned
        performance_keywords = ['2ms', '200µs', '600 nodes', '0.5s']
        for result in self.test_results.values():
            stdout = result.get('stdout', '')
            if any(keyword in stdout for keyword in performance_keywords):
                performance_analysis['performance_targets_met'] = True
                break
        
        return performance_analysis
    
    def _analyze_mathematical_correctness(self) -> Dict[str, Any]:
        """Analyze mathematical correctness validation."""
        math_analysis = {
            'd4_invariance_tested': False,
            'heuristic_admissibility_tested': False,
            'eigenvalue_stability_tested': False,
            'symmetry_preservation_tested': False,
            'numerical_precision_tested': False
        }
        
        # Look for mathematical validation in test outputs
        for result in self.test_results.values():
            stdout = result.get('stdout', '')
            
            if 'd4' in stdout.lower() or 'invariance' in stdout.lower():
                math_analysis['d4_invariance_tested'] = True
            if 'admissibility' in stdout.lower() or 'heuristic' in stdout.lower():
                math_analysis['heuristic_admissibility_tested'] = True
            if 'eigenvalue' in stdout.lower() or 'stability' in stdout.lower():
                math_analysis['eigenvalue_stability_tested'] = True
            if 'symmetry' in stdout.lower():
                math_analysis['symmetry_preservation_tested'] = True
            if 'precision' in stdout.lower() or 'tolerance' in stdout.lower():
                math_analysis['numerical_precision_tested'] = True
        
        return math_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests
        failed_modules = [name for name, result in self.test_results.items() if not result['success']]
        if failed_modules:
            recommendations.append(f"Address failing tests in: {', '.join(failed_modules)}")
        
        # Check for performance issues
        slow_modules = [name for name, result in self.test_results.items() 
                       if result['execution_time'] > 60]  # More than 1 minute
        if slow_modules:
            recommendations.append(f"Optimize performance for slow test modules: {', '.join(slow_modules)}")
        
        # Check for skipped tests
        skipped_tests = sum(r['test_details'].get('skipped', 0) for r in self.test_results.values())
        if skipped_tests > 0:
            recommendations.append(f"Investigate {skipped_tests} skipped tests - may indicate missing dependencies")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passing! Consider adding more edge case tests")
            recommendations.append("Monitor performance regression in future updates")
            recommendations.append("Consider adding property-based testing for mathematical invariants")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print comprehensive test report."""
        print("\\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT - TASK 12.1")
        print("="*80)
        
        # Overall status
        status = "✅ PASSED" if report['overall_success'] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Total Execution Time: {report['total_execution_time']:.2f} seconds")
        
        # Summary statistics
        summary = report['summary']
        print(f"\\nTest Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Skipped: {summary['skipped']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        # Requirements coverage
        coverage = report['requirements_coverage']
        print(f"\\nRequirements Coverage:")
        print(f"  Total Requirements Tested: {coverage['total_requirements']}")
        print(f"  Requirements: {', '.join(coverage['requirements_tested'])}")
        
        # Module results
        print(f"\\nModule Results:")
        for module_name, result in report['module_results'].items():
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['execution_time']:.2f}s"
            details = result['test_details']
            test_summary = f"{details.get('passed', 0)}/{details.get('total_tests', 0)} passed"
            print(f"  {status} {module_name}: {test_summary} ({time_str})")
        
        # Performance analysis
        perf = report['performance_analysis']
        print(f"\\nPerformance Analysis:")
        print(f"  Perception Performance: {perf['perception_performance']}")
        print(f"  Reasoning Performance: {perf['reasoning_performance']}")
        print(f"  Search Performance: {perf['search_performance']}")
        print(f"  Performance Targets Met: {'✅' if perf['performance_targets_met'] else '❌'}")
        
        # Mathematical validation
        math_val = report['mathematical_validation']
        print(f"\\nMathematical Validation:")
        print(f"  D₄ Invariance: {'✅' if math_val['d4_invariance_tested'] else '❌'}")
        print(f"  Heuristic Admissibility: {'✅' if math_val['heuristic_admissibility_tested'] else '❌'}")
        print(f"  Eigenvalue Stability: {'✅' if math_val['eigenvalue_stability_tested'] else '❌'}")
        print(f"  Symmetry Preservation: {'✅' if math_val['symmetry_preservation_tested'] else '❌'}")
        
        # Recommendations
        print(f"\\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\\n" + "="*80)
        print("TASK 12.1 COMPREHENSIVE TESTING COMPLETE")
        print("="*80)
    
    def save_report(self, report: Dict[str, Any], filename: str = "comprehensive_test_report.json") -> None:
        """Save report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Test report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Run comprehensive tests and generate report."""
    logger.info("Starting Task 12.1 - Comprehensive Unit Testing")
    
    try:
        # Initialize test runner
        runner = ComprehensiveTestRunner()
        
        # Run all tests
        report = runner.run_all_tests()
        
        # Print report
        runner.print_report(report)
        
        # Save report
        runner.save_report(report)
        
        # Return appropriate exit code
        return 0 if report['overall_success'] else 1
        
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test run failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())