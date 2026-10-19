# Sourced by the experiment scripts.
#
# compare MGF INSTRUMENT NAME [predict options...]
#   Run BUDDY and SIRIUS on MGF (scripts/run_buddy_sirius.py), then predict
#   with FIDDLE alone, FIDDLE + BUDDY, FIDDLE + BUDDY + SIRIUS, and all three
#   with integrated scoring, writing ./result/{fiddle,fiddle_buddy,
#   fiddle_buddy_sirius,integrated}_NAME.csv with --eval.
compare() {
    local mgf=$1 inst=$2 name=$3
    shift 3
    local ext=./external/$name
    local predict="python -m msfiddle predict --test_data $mgf --instrument_type $inst --num_workers 8 --eval $*"

    python scripts/run_buddy_sirius.py --mgf "$mgf" --instrument_type "$inst" --out_dir "$ext"

    $predict --result_path ./result/fiddle_$name.csv
    if [ -d "$ext/buddy" ]; then
        $predict --buddy_path "$ext/buddy" --result_path ./result/fiddle_buddy_$name.csv
    fi
    if [ -d "$ext/buddy" ] && [ -d "$ext/sirius" ]; then
        $predict --buddy_path "$ext/buddy" --sirius_path "$ext/sirius" \
            --result_path ./result/fiddle_buddy_sirius_$name.csv
        $predict --buddy_path "$ext/buddy" --sirius_path "$ext/sirius" --integrate \
            --result_path ./result/integrated_$name.csv
    fi
}
