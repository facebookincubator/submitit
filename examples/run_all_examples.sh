#!/bin/bash

# Run all submitthem examples
# This script demonstrates submitthem functionality across all scheduler types

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running submitthem examples from: $SCRIPT_DIR"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_example() {
    local example_path="$1"
    local example_name="$(basename "$example_path")"
    local scheduler="$(basename "$(dirname "$example_path")")"
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Running: $scheduler/$example_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if python "$example_path"; then
        echo -e "${GREEN}✓ $scheduler/$example_name completed successfully${NC}"
    else
        echo -e "${YELLOW}⚠ $scheduler/$example_name failed or not available${NC}"
    fi
    echo ""
}

# Track results
declare -a PASSED
declare -a FAILED

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          SUBMITTHEM EXAMPLES RUNNER${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Run local examples
echo -e "${YELLOW}▶ LOCAL EXECUTOR EXAMPLES${NC}"
echo ""
for example in "$SCRIPT_DIR/local"/*.py; do
    if [ -f "$example" ]; then
        if run_example "$example"; then
            PASSED+=("local/$(basename "$example")")
        else
            FAILED+=("local/$(basename "$example")")
        fi
    fi
done

# Run SLURM examples (if SLURM available)
echo -e "${YELLOW}▶ SLURM EXECUTOR EXAMPLES${NC}"
echo ""
if command -v sbatch &> /dev/null; then
    for example in "$SCRIPT_DIR/slurm"/*.py; do
        if [ -f "$example" ]; then
            if run_example "$example"; then
                PASSED+=("slurm/$(basename "$example")")
            else
                FAILED+=("slurm/$(basename "$example")")
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠ SLURM not available (sbatch not found), skipping SLURM examples${NC}"
    echo ""
fi

# Run PBS examples (if PBS available)
echo -e "${YELLOW}▶ PBS EXECUTOR EXAMPLES${NC}"
echo ""
if command -v qsub &> /dev/null; then
    for example in "$SCRIPT_DIR/pbs"/*.py; do
        if [ -f "$example" ]; then
            if run_example "$example"; then
                PASSED+=("pbs/$(basename "$example")")
            else
                FAILED+=("pbs/$(basename "$example")")
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠ PBS not available (qsub not found), skipping PBS examples${NC}"
    echo ""
fi

# Print summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}Passed (${#PASSED[@]}):${NC}"
for example in "${PASSED[@]}"; do
    echo "  ✓ $example"
done
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${YELLOW}Failed (${#FAILED[@]}):${NC}"
    for example in "${FAILED[@]}"; do
        echo "  ✗ $example"
    done
    echo ""
fi

echo -e "${BLUE}Total: $((${#PASSED[@]} + ${#FAILED[@]})) examples${NC}"
echo ""

# Exit with status
if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All available examples completed${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some examples failed (this may be expected if schedulers are unavailable)${NC}"
    exit 1
fi
