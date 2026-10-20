# Sourced by the experiment scripts.
#
# compare MGF INSTRUMENT NAME [predict options...]
#   Run BUDDY and SIRIUS on MGF (scripts/run_buddy_sirius.py), then predict
#   with FIDDLE alone, FIDDLE + BUDDY and FIDDLE + BUDDY + SIRIUS (their
#   candidates pooled and rescored by FIDDLE), writing
#   ./result/{fiddle,fiddle_buddy,fiddle_buddy_sirius}_NAME.csv with --eval.
#   Env: MODEL_TAG uses ./check_point/fiddle_{tcn,rescore}_INSTRUMENT_MODEL_TAG.pt
#   (e.g. 102026_muon, from train_release_v2.3.sh) instead of the released
#   models; a --resume_path in the options still takes precedence.
compare() {
    local mgf=$1 inst=$2 name=$3
    shift 3
    local ext=./external/$name
    local models=""
    [ -n "$MODEL_TAG" ] && models="--resume_path ./check_point/fiddle_tcn_${inst}_$MODEL_TAG.pt --rescore_resume_path ./check_point/fiddle_rescore_${inst}_$MODEL_TAG.pt"
    local predict="python -m msfiddle predict --test_data $mgf --instrument_type $inst --num_workers 8 --eval $models $*"

    python scripts/run_buddy_sirius.py --mgf "$mgf" --instrument_type "$inst" --out_dir "$ext"

    $predict --result_path ./result/fiddle_$name.csv
    if [ -d "$ext/buddy" ]; then
        $predict --buddy_path "$ext/buddy" --result_path ./result/fiddle_buddy_$name.csv
    fi
    if [ -d "$ext/buddy" ] && [ -d "$ext/sirius" ]; then
        $predict --buddy_path "$ext/buddy" --sirius_path "$ext/sirius" \
            --result_path ./result/fiddle_buddy_sirius_$name.csv
    fi
}
