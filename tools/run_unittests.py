#!/usr/bin/env python
import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="src/tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    num_failures = len(result.failures)
    num_errors = len(result.errors)
    num_skipped = len(result.skipped)
    num_expected_failures = len(result.expectedFailures)
    num_unexpected_successes = len(result.unexpectedSuccesses)

    # Count passes as successful runs excluding skipped and expected failures
    num_passed = total - (num_failures + num_errors + num_skipped + num_expected_failures)

    print("\n==================== SUMMARY ====================")
    print(
        f"total={total}, passed={num_passed}, failures={num_failures}, "
        f"errors={num_errors}, skipped={num_skipped}, xfail={num_expected_failures}, xpass={num_unexpected_successes}"
    )
    print("================================================\n")

    # Non-zero exit if failures or errors
    sys.exit(1 if (num_failures > 0 or num_errors > 0) else 0)


if __name__ == "__main__":
    main()


