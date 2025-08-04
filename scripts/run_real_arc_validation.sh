#!/bin/bash
# Script to run real ARC dataset validation

# Set the path to your ARC dataset
ARC_DATASET_PATH="/Users/mazalcohen/Downloads/arc-prize-2025 2"

# Check if dataset path exists
if [ ! -d "$ARC_DATASET_PATH" ]; then
    echo "❌ ARC dataset not found at: $ARC_DATASET_PATH"
    echo "Please update the ARC_DATASET_PATH variable in this script"
    exit 1
fi

echo "🚀 Starting Real ARC Dataset Validation"
echo "Dataset path: $ARC_DATASET_PATH"
echo "=================================="

# Run validation with different configurations
echo "📊 Running validation on first 50 tasks (quick test)..."
python3 scripts/validate_real_arc_dataset.py "$ARC_DATASET_PATH" \
    --max-tasks 50 \
    --timeout 8.0 \
    --output "real_arc_validation_50_tasks.json"

echo ""
echo "📊 Running validation on first 100 tasks (medium test)..."
python3 scripts/validate_real_arc_dataset.py "$ARC_DATASET_PATH" \
    --max-tasks 100 \
    --timeout 10.0 \
    --output "real_arc_validation_100_tasks.json"

echo ""
echo "📊 Running full validation on all tasks (complete test)..."
python3 scripts/validate_real_arc_dataset.py "$ARC_DATASET_PATH" \
    --timeout 15.0 \
    --output "real_arc_validation_full.json"

echo ""
echo "✅ Real ARC validation complete!"
echo "Check the generated JSON files for detailed results."