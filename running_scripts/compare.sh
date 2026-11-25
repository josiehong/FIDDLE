# Sourced by the experiment scripts.
#
# compare MGF INSTRUMENT NAME [predict options...]
#   Run BUDDY and SIRIUS on MGF, then FIDDLE alone, with BUDDY, and with BUDDY
#   and SIRIUS, writing ./result/{fiddle,fiddle_buddy,fiddle_buddy_sirius}_NAME.csv.
#   FIDDLE_ONLY=1 runs FIDDLE alone, without BUDDY and SIRIUS.
#   MODEL_TAG uses ./check_point/fiddle_{tcn,rescore}_INSTRUMENT_MODEL_TAG.pt
#   instead of the released models.

# SIRIUS installed by install_sirius.sh
export PATH="$PWD/tools/sirius/bin:$PWD/tools/sirius.app/Contents/MacOS:$PATH"

compare() {
    local mgf=$1 inst=$2 name=$3
    shift 3
    local ext=./external/$name
    local models=""
    [ -n "$MODEL_TAG" ] && models="--resume_path ./check_point/fiddle_tcn_${inst}_$MODEL_TAG.pt --rescore_resume_path ./check_point/fiddle_rescore_${inst}_$MODEL_TAG.pt"
    local predict="python -m msfiddle predict --test_data $mgf --instrument_type $inst --num_workers 8 --eval $models $*"

    # Drop this benchmark's earlier results, so a step that does not run
    # leaves no stale file for plot_benchmarks.py to pick up
    local method suffix
    for method in fiddle fiddle_buddy fiddle_buddy_sirius; do
        for suffix in .csv _eval.csv _eval_details.csv _eval_groups.csv; do
            rm -f "./result/${method}_$name$suffix"
        done
    done

    if [ -n "$FIDDLE_ONLY" ]; then
        $predict --result_path ./result/fiddle_$name.csv
        return
    fi
    python scripts/run_buddy_sirius.py --mgf "$mgf" --instrument_type "$inst" --out_dir "$ext"
    # run_buddy_sirius.py skips a tool that is not installed
    local tool
    for tool in buddy sirius; do
        if [ ! -d "$ext/$tool" ]; then
            echo "No $tool results in $ext: install it (running_scripts/install_$tool.sh) or set FIDDLE_ONLY=1"
            return 1
        fi
    done

    $predict --result_path ./result/fiddle_$name.csv
    $predict --buddy_path "$ext/buddy" --result_path ./result/fiddle_buddy_$name.csv
    $predict --buddy_path "$ext/buddy" --sirius_path "$ext/sirius" \
        --result_path ./result/fiddle_buddy_sirius_$name.csv
}
