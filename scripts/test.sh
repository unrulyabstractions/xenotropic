#!/bin/bash
# Run all tests for xenotropic
#
# Usage:
#   ./scripts/test.sh          # Run all tests
#   ./scripts/test.sh -v       # Run with verbose output
#   ./scripts/test.sh -x       # Stop on first failure
#   ./scripts/test.sh --cov    # Run with coverage report
#   ./scripts/test.sh -k "test_string"  # Run matching tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}                  XENOTROPIC TEST SUITE                     ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    if [ -f ".venv/bin/pytest" ]; then
        PYTEST=".venv/bin/pytest"
    else
        echo -e "${RED}Error: pytest not found. Install with: pip install pytest${NC}"
        exit 1
    fi
else
    PYTEST="pytest"
fi

# Default pytest options
PYTEST_OPTS=""

# Parse coverage option
COVERAGE=""
if [[ " $@ " =~ " --cov " ]]; then
    COVERAGE="--cov=xenotechnics --cov=exploration --cov-report=term-missing --cov-report=html"
    # Remove --cov from args
    set -- "${@/--cov/}"
fi

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Project root: $PROJECT_ROOT"
echo "  Pytest: $PYTEST"
echo "  Extra args: $@"
if [ -n "$COVERAGE" ]; then
    echo "  Coverage: enabled"
fi
echo ""

# Run tests by category
run_tests() {
    local category=$1
    local path=$2

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Running: $category${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if $PYTEST "$path" $COVERAGE "$@"; then
        echo -e "${GREEN}✓ $category passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $category failed${NC}"
        return 1
    fi
}

# Track test results
FAILED=0

echo -e "${YELLOW}Running all tests...${NC}"
echo ""

# Run all tests at once for efficiency
if $PYTEST tests/ $COVERAGE "$@"; then
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}                    ALL TESTS PASSED                        ${NC}"
    echo -e "${GREEN}============================================================${NC}"
else
    FAILED=1
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}                    SOME TESTS FAILED                       ${NC}"
    echo -e "${RED}============================================================${NC}"
fi

# Print summary
echo ""
echo -e "${YELLOW}Test Categories:${NC}"
echo "  • xenotechnics/common  - String, Structure, System abstractions"
echo "  • xenotechnics/trees   - TreeNode, LLMTree for generation"
echo "  • xenotechnics/structures - Classifier, Similarity, Grammar structures"
echo "  • xenotechnics/operators  - Vector and Entropic operators"
echo "  • xenotechnics/systems    - VectorSystem and compliance"
echo "  • xenotechnics/dynamics   - Linear dynamics"
echo "  • exploration          - Generation strategies"
echo "  • integration          - End-to-end tests"
echo ""

if [ -n "$COVERAGE" ]; then
    echo -e "${YELLOW}Coverage report:${NC}"
    echo "  HTML report: htmlcov/index.html"
    echo ""
fi

# Tips
echo -e "${YELLOW}Tips:${NC}"
echo "  Run specific tests:    ./scripts/test.sh -k 'test_string'"
echo "  Run with verbose:      ./scripts/test.sh -v"
echo "  Stop on first fail:    ./scripts/test.sh -x"
echo "  Run with coverage:     ./scripts/test.sh --cov"
echo "  Run single file:       ./scripts/test.sh tests/xenotechnics/common/test_string.py"
echo ""

exit $FAILED
