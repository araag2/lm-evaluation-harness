#!/bin/bash
# ================================================
# Error Breakdown Extraction Script
# ================================================
# Usage examples:
#   ./extract_errors.sh --input ./outputs
#   ./extract_errors.sh --input ./outputs --output ./outputs/error_analysis
#   ./extract_errors.sh --input-list "dir1,dir2" --name detailed_errors

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
INPUT_FOLDER="./outputs"
OUTPUT_DIR="./outputs/error_breakdown"
OUTPUT_NAME="error_breakdown"
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|--input-folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --name|--output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << EOF
Error Breakdown Extraction Script

Usage: $0 [OPTIONS]

Input Options:
  --input DIR                      Input directory containing evaluation results
                                   (default: ./outputs)

Output Options:
  --output DIR                     Output directory (default: ./outputs/error_breakdown)
  --name NAME                      Output filename prefix (default: error_breakdown)

Other Options:
  --dry-run                        Show what would be extracted without executing
  --help, -h                       Show this help message

Examples:
  # Extract errors from default outputs directory
  $0

  # Extract from specific directory
  $0 --input ./outputs/multi-turn_CoT

  # Custom output location and name
  $0 --input ./outputs --output ./analysis --name experiment_errors

  # Preview extraction
  $0 --input ./outputs --dry-run

The script will create:
  - {output_dir}/{output_name}.json       (Detailed error breakdown)
  - {output_dir}/aggregated_errors.json   (Aggregated statistics)

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate input
if [ ! -d "$INPUT_FOLDER" ]; then
    log_error "Input folder not found: $INPUT_FOLDER"
    exit 1
fi

# Show configuration
print_separator
log_info "Error Breakdown Extraction Configuration"
print_separator
echo "Input Folder:    $INPUT_FOLDER"
echo "Output Dir:      $OUTPUT_DIR"
echo "Output Name:     $OUTPUT_NAME"
echo "Dry Run:         $DRY_RUN"
print_separator

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode: No extraction will be executed"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Execute extraction
log_info "Running error extraction script..."
echo ""

python "${SCRIPT_DIR}/scripts/extract_error-breakdown.py" \
    --input_folder "$INPUT_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME"

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    log_success "Error extraction completed successfully"
    print_separator
    log_info "Output files:"
    echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.json"
    if [ -f "${OUTPUT_DIR}/aggregated_errors.json" ]; then
        echo "  - ${OUTPUT_DIR}/aggregated_errors.json"
    fi
    print_separator
else
    log_error "Error extraction failed with exit code: $STATUS"
    exit $STATUS
fi
