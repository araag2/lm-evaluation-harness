#!/bin/bash
# ================================================
# Vote-Only Evaluation Runner
# ================================================
# Usage examples:
#   ./run_vote.sh --vote-file path/to/summary.json
#   ./run_vote.sh --vote-dir ./outputs/multi-turn_CoT --pattern "Summary*.json"
#   ./run_vote.sh --vote-list "file1.json,file2.json"

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
PROVIDER="vllm"
VOTE_FILES=()
OUTPUT_BASE="./outputs/only-vote"
CUDA_DEVICES="0"
BATCH_SIZE="auto"
SEED="0"
DRY_RUN=false
USE_TIMESTAMP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vote-file)
            VOTE_FILES+=("$2")
            shift 2
            ;;
        --vote-list)
            IFS=',' read -ra FILE_LIST <<< "$2"
            VOTE_FILES+=("${FILE_LIST[@]}")
            shift 2
            ;;
        --vote-dir)
            VOTE_DIR="$2"
            PATTERN="${3:-Summary*.json}"
            if [ -d "$VOTE_DIR" ]; then
                while IFS= read -r -d '' file; do
                    VOTE_FILES+=("$file")
                done < <(find "$VOTE_DIR" -name "$PATTERN" -type f -print0)
            else
                log_error "Directory not found: $VOTE_DIR"
                exit 1
            fi
            shift 2
            ;;
        --pattern)
            # This is consumed by --vote-dir, skip if standalone
            shift
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --gpu|--cuda)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --timestamp)
            USE_TIMESTAMP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << EOF
Vote-Only Evaluation Runner

Usage: $0 [OPTIONS]

Vote File Selection:
  --vote-file FILE                 Specify a single vote file
  --vote-list FILE1,FILE2,...      Comma-separated list of vote files
  --vote-dir DIR [PATTERN]         Find all vote files in directory matching pattern
                                   (default pattern: Summary*.json)

Evaluation Options:
  --output PATH                    Base output directory (default: ./outputs/only-vote)
  --gpu ID                         CUDA device ID (default: 0)
  --batch-size SIZE                Batch size (default: auto)
  --seed SEED                      Random seed (default: 0)

Other Options:
  --timestamp                      Add timestamp to output paths
  --dry-run                        Show what would run without executing
  --help, -h                       Show this help message

Examples:
  # Run vote on single file
  $0 --vote-file ./outputs/multi-turn_CoT/MedQA_CoT/qwen3-4b/Summary_2025-01-15T10-30.json

  # Run vote on multiple files
  $0 --vote-list "file1.json,file2.json,file3.json"

  # Find and run vote on all summary files in a directory
  $0 --vote-dir ./outputs/multi-turn_CoT --pattern "Summary*.json"

  # Dry-run to see which files would be processed
  $0 --vote-dir ./outputs/multi-turn_CoT --dry-run

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
if [ ${#VOTE_FILES[@]} -eq 0 ]; then
    log_error "No vote files specified. Use --vote-file, --vote-list, or --vote-dir"
    exit 1
fi

# Check GPU availability
check_gpu "$CUDA_DEVICES"

# Calculate total runs
TOTAL_RUNS=${#VOTE_FILES[@]}
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Show configuration
print_separator
log_info "Vote-Only Evaluation Configuration"
print_separator
echo "Provider:        $PROVIDER"
echo "Vote Files:      ${#VOTE_FILES[@]}"
echo "Output Base:     $OUTPUT_BASE"
echo "GPU:             $CUDA_DEVICES"
echo "Batch Size:      $BATCH_SIZE"
echo "Seed:            $SEED"
echo "Dry Run:         $DRY_RUN"
print_separator

# Show files to be processed
log_info "Vote files to process:"
for file in "${VOTE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $(basename "$file")"
    else
        echo "  ✗ $(basename "$file") (NOT FOUND)"
    fi
done
print_separator

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode: No evaluations will be executed"
    exit 0
fi

# Create output directory
if [ "$USE_TIMESTAMP" = true ]; then
    OUTPUT_PATH=$(create_output_dir "$OUTPUT_BASE" true)
else
    OUTPUT_PATH="$OUTPUT_BASE"
    mkdir -p "$OUTPUT_PATH"
fi

# Record start time
START_TIME=$(date +%s)

# Main execution loop
for VOTE_FILE in "${VOTE_FILES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Check if file exists
    if [ ! -f "$VOTE_FILE" ]; then
        log_error "File not found: $VOTE_FILE"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        continue
    fi
    
    show_progress "$CURRENT_RUN" "$TOTAL_RUNS" "$(basename "$VOTE_FILE")"
    
    if run_vote_only "$PROVIDER" "$VOTE_FILE" "$OUTPUT_PATH" \
                     "$BATCH_SIZE" "$SEED" "$CUDA_DEVICES"; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        log_warning "Continuing with next file..."
    fi
    
    echo ""
done

# Record end time
END_TIME=$(date +%s)

# Show summary
show_summary "$TOTAL_RUNS" "$SUCCESSFUL_RUNS" "$FAILED_RUNS" "$START_TIME" "$END_TIME"

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
