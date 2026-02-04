#!/bin/bash
# ================================================
# Results Extraction Script
# ================================================
# Usage examples:
#   ./extract_results.sh --input ./outputs --output ./outputs/collected_results
#   ./extract_results.sh --input-list "dir1,dir2,dir3" --format csv
#   ./extract_results.sh --input ./outputs --recursive --name my_results

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
INPUT_FOLDERS=()
OUTPUT_DIR="./outputs/collected_results"
OUTPUT_NAME="result_summary"
RECURSIVE=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FOLDERS+=("$2")
            shift 2
            ;;
        --input-list)
            IFS=',' read -ra FOLDER_LIST <<< "$2"
            INPUT_FOLDERS+=("${FOLDER_LIST[@]}")
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
        --recursive)
            RECURSIVE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << EOF
Results Extraction Script

Usage: $0 [OPTIONS]

Input Options:
  --input DIR                      Input directory containing results
  --input-list DIR1,DIR2,...       Comma-separated list of input directories
  --recursive                      Search subdirectories recursively

Output Options:
  --output DIR                     Output directory (default: ./outputs/collected_results)
  --name NAME                      Output filename prefix (default: result_summary)

Other Options:
  --dry-run                        Show what would be extracted without executing
  --help, -h                       Show this help message

Examples:
  # Extract from single directory
  $0 --input ./outputs/multi-turn_CoT

  # Extract from multiple directories
  $0 --input-list "./outputs/multi-turn_CoT,./outputs/multi-turn_SC-CoT"

  # Extract recursively with custom output name
  $0 --input ./outputs --recursive --name full_benchmark_results

  # Preview extraction
  $0 --input ./outputs --dry-run

The script will create:
  - {output_dir}/{output_name}.csv    (CSV format)
  - {output_dir}/{output_name}.json   (JSON format)
  - {output_dir}/{output_name}.xlsx   (Excel format, if openpyxl available)

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

# Validate inputs
if [ ${#INPUT_FOLDERS[@]} -eq 0 ]; then
    log_error "No input folders specified. Use --input or --input-list"
    exit 1
fi

# Show configuration
print_separator
log_info "Results Extraction Configuration"
print_separator
echo "Input Folders:   ${#INPUT_FOLDERS[@]}"
for folder in "${INPUT_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        echo "  ✓ $folder"
    else
        echo "  ✗ $folder (NOT FOUND)"
    fi
done
echo "Output Dir:      $OUTPUT_DIR"
echo "Output Name:     $OUTPUT_NAME"
echo "Recursive:       $RECURSIVE"
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
log_info "Running extraction script..."
echo ""

python "${SCRIPT_DIR}/scripts/extract_run-results.py" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --input_folders "${INPUT_FOLDERS[@]}"

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    log_success "Results extraction completed successfully"
    print_separator
    log_info "Output files:"
    echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.json"
    echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.csv"
    if [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.xlsx" ]; then
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.xlsx"
    fi
    print_separator
else
    log_error "Results extraction failed with exit code: $STATUS"
    exit $STATUS
fi
