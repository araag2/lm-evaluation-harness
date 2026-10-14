#!/bin/bash
# ================================================
# Results Extraction Script
# ================================================
# Usage examples:
#   bash extract_results.sh --input-list "dir1,dir2,dir3" --output ./outputs/res
#   bash extract_results.sh --input ./outputs --output ./outputs/res --no-charts

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
INPUT_FOLDERS=()
OUTPUT_DIR="./outputs/collected_results"
OUTPUT_NAME="result_summary"
FILE_FILTER=""
NO_CSV=false
NO_MARKDOWN=false
NO_LATEX=false
NO_CHARTS=false
ALL_RUNS=false
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
        --file-filter)
            FILE_FILTER="$2"
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
        --no-latex)
            NO_LATEX=true
            shift
            ;;
        --no-charts)
            NO_CHARTS=true
            shift
            ;;
        --all-runs)
            ALL_RUNS=true
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
  --file-filter PATTERN            Only process JSON files containing this pattern

Output Options:
  --output DIR                     Output directory (default: ./outputs/collected_results)
  --name NAME                      Output filename prefix (default: result_summary)
  --no-csv                         Skip CSV output generation
  --no-markdown                    Skip Markdown output generation
  --no-latex                       Skip LaTeX output generation
  --no-charts                      Skip chart/visualization generation
  --all-runs                       Show every run, not one per task, mode and model

Other Options:
  --dry-run                        Show what would be extracted without executing
  --help, -h                       Show this help message

Examples:
  # Extract from single directory
  $0 --input ./outputs/multi-turn_CoT

  # Extract from multiple directories
  $0 --input-list "./outputs/multi-turn_CoT,./outputs/multi-turn_SC-CoT"

  # Extract with custom output name and filter
  $0 --input ./outputs --name full_benchmark_results --file-filter "Qwen"

  # Extract without charts
  $0 --input ./outputs --no-charts

  # Preview extraction
  $0 --input ./outputs --dry-run

By default one run is kept per task, mode and model: the one on most samples, then the latest.

The script will create (unless disabled):
  - {output_dir}/{output_name}.csv              (one row per task, mode, model, metric and slice)
  - {output_dir}/{output_name}.md               (overview per mode, details per dataset)
  - {output_dir}/{output_name}_sources.md       (file of each score, runs not shown)
  - {output_dir}/{output_name}.tex              (the same tables for LaTeX)
  - {output_dir}/charts/, heatmaps/             (charts and heatmaps per dataset)

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
if [ -n "$FILE_FILTER" ]; then
    echo "File Filter:     $FILE_FILTER"
fi
echo "Skip CSV:        $NO_CSV"
echo "Skip Markdown:   $NO_MARKDOWN"
echo "Skip LaTeX:      $NO_LATEX"
echo "Skip Charts:     $NO_CHARTS"
echo "All Runs:        $ALL_RUNS"
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

PYTHON_ARGS=(--output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME" --input_folders "${INPUT_FOLDERS[@]}")
[ -n "$FILE_FILTER" ] && PYTHON_ARGS+=(--file_filter "$FILE_FILTER")
[ "$NO_CSV" = true ] && PYTHON_ARGS+=(--no-csv)
[ "$NO_MARKDOWN" = true ] && PYTHON_ARGS+=(--no-markdown)
[ "$NO_LATEX" = true ] && PYTHON_ARGS+=(--no-latex)
[ "$NO_CHARTS" = true ] && PYTHON_ARGS+=(--no-charts)
[ "$ALL_RUNS" = true ] && PYTHON_ARGS+=(--all-runs)

STATUS=0
python "${SCRIPT_DIR}/../scripts/extract_run-results.py" "${PYTHON_ARGS[@]}" || STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    log_success "Results extraction completed successfully"
    print_separator
    log_info "Output files:"
    
    [ "$NO_CSV" = false ] && [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.csv" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.csv"
    
    [ "$NO_MARKDOWN" = false ] && [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.md" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.md"
    
    [ "$NO_LATEX" = false ] && [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.tex" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}.tex"
    
    [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}_sources.md" ] && \
        echo "  - ${OUTPUT_DIR}/${OUTPUT_NAME}_sources.md"

    if [ "$NO_CHARTS" = false ]; then
        echo "  - $(ls "${OUTPUT_DIR}/charts" 2>/dev/null | wc -l) chart(s) in ${OUTPUT_DIR}/charts"
        echo "  - $(ls "${OUTPUT_DIR}/heatmaps" 2>/dev/null | wc -l) heatmap(s) in ${OUTPUT_DIR}/heatmaps"
    fi

    print_separator
else
    log_error "Results extraction failed with exit code: $STATUS"
    exit $STATUS
fi
