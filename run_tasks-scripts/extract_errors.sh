#!/bin/bash
# ================================================
# Error Breakdown Extraction Script
# ================================================
# Usage examples:
#   bash extract_errors.sh --input ./outputs
#   bash extract_errors.sh --input ./outputs --output ./outputs/error_analysis
#   bash extract_errors.sh --input-list "dir1,dir2" --name detailed_errors

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
INPUT_FOLDER="./outputs"
OUTPUT_DIR="./outputs/error_breakdown"
OUTPUT_NAME="error_breakdown"
USE_PREDICTION="majority"
NO_CSV=false
NO_MARKDOWN=false
NO_CHARTS=false
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
        --use)
            USE_PREDICTION="$2"
            shift 2
            ;;
        --no-csv)
            NO_CSV=true
            shift
            ;;
        --no-markdown)
            NO_MARKDOWN=true
            shift
            ;;
        --no-charts)
            NO_CHARTS=true
            shift
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
  --no-csv                         Skip CSV output generation
  --no-markdown                    Skip Markdown output generation
  --no-charts                      Skip chart/visualization generation

Analysis Options:
  --use PREDICTION                 Prediction key to use (default: majority)

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

  # Skip visualizations
  $0 --input ./outputs --no-charts

  # Preview extraction
  $0 --input ./outputs --dry-run

The script will create (unless disabled):
  - {output_dir}/{output_name}.json              (Detailed error breakdown)
  - {output_dir}/{output_name}.csv               (CSV format with error stats)
  - {output_dir}/{output_name}.md                (Markdown format with tables per dataset)
  - {output_dir}/{output_name}_summary.txt       (Text summary report)
  - {output_dir}/{output_name}_*_accuracy.png    (Accuracy charts per dataset)

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
echo "Use Prediction:  $USE_PREDICTION"
echo "Skip CSV:        $NO_CSV"
echo "Skip Markdown:   $NO_MARKDOWN"
echo "Skip Charts:     $NO_CHARTS"
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

# Build Python command with all arguments
PYTHON_CMD="${SCRIPT_DIR}/../scripts/extract_error-breakdown.py \
    --input_folder \"$INPUT_FOLDER\" \
    --output_dir \"$OUTPUT_DIR\" \
    --output_name \"$OUTPUT_NAME\" \
    --use \"$USE_PREDICTION\""

if [ "$NO_CSV" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-csv"
fi

if [ "$NO_MARKDOWN" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-markdown"
fi

if [ "$NO_CHARTS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-charts"
fi

eval python $PYTHON_CMD

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    log_success "Error extraction completed successfully"
    print_separator
    log_info "Output files:"
    
    [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.json" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.json"
    
    [ "$NO_CSV" = false ] && [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.csv" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.csv"
    
    [ "$NO_MARKDOWN" = false ] && [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.md" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.md"
    
    [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}_summary.txt" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}_summary.txt"
    
    if [ "$NO_CHARTS" = false ]; then
        # Count chart files
        CHART_COUNT=$(ls "${OUTPUT_DIR}/${OUTPUT_NAME}"_*_accuracy.png 2>/dev/null | wc -l)
        if [ "$CHART_COUNT" -gt 0 ]; then
            echo "  - ${CHART_COUNT} accuracy chart(s)"
        fi
    fi
    
    print_separator
else
    log_error "Error extraction failed with exit code: $STATUS"
    exit $STATUS
fi
