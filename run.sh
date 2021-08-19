#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

function run_exp1 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 24 -f 0 --attack NA --LT"
    for seed in 0 1 2
    do
        for s in 0 2
        do
            for agg in "cm" "cp" "rfa" "krum" "avg"
            do
                python exp1.py $COMMON_OPTIONS --agg $agg --bucketing $s --seed $seed &
                pids[$!]=$!

                python exp1.py $COMMON_OPTIONS --agg $agg --noniid --bucketing $s --seed $seed &
                pids[$!]=$!
            done

            # wait for all pids
            for pid in ${pids[*]}; do
                wait $pid
            done
            unset pids
        done
    done
}

function run_exp1_addon {
    # TODO: Delete the function after the experiment is done.
    # This part of exp1 is redone because of a bug.
    COMMON_OPTIONS="--use-cuda --identifier all -n 24 -f 0 --attack NA --LT"
    for seed in 0 1 2
    do
        python exp1.py $COMMON_OPTIONS --agg "krum" --bucketing 2 --seed $seed &
        pids[$!]=$!

        python exp1.py $COMMON_OPTIONS --agg "krum" --noniid --bucketing 2 --seed $seed &
        pids[$!]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function exp1_plot {
    COMMON_OPTIONS="--use-cuda --identifier all -n 24 -f 0 --attack NA --LT"
    python exp1.py $COMMON_OPTIONS --plot
}

function run_exp2 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --attack mimic"
    for seed in 0 1 2
    do
        for agg in "cm" "avg" "cp" "rfa" "krum" 
        do
            for s in 0 2
            do
                python exp2.py $COMMON_OPTIONS --agg $agg --bucketing $s --seed $seed &
                pids[$!]=$!

                python exp2.py $COMMON_OPTIONS --agg $agg --noniid --bucketing $s --seed $seed &
                pids[$!]=$!
            done

            # wait for all pids
            for pid in ${pids[*]}; do
                wait $pid
            done
            unset pids
        done
    done
}

function run_exp2_addon {
    # TODO: Delete the function after the experiment is done.
    # This part of exp1 is redone because of a bug.
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --attack mimic"
    for seed in 0 1 2
    do
        python exp2.py $COMMON_OPTIONS --agg "krum" --bucketing 2 --seed $seed &
        pids[$!]=$!

        python exp2.py $COMMON_OPTIONS --agg "krum" --noniid --bucketing 2 --seed $seed &
        pids[$!]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}


function run_exp3 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
    for seed in 0 1 2
    do
        for atk in "BF" "LF" "mimic" "IPM" "ALIE"
        do
            for s in 0 2
            do
                for m in 0 0.9
                do
                    for agg in "cm" "cp" "rfa" "krum" 
                    do
                        python exp3.py $COMMON_OPTIONS --attack $atk --agg $agg --bucketing $s --seed $seed --momentum $m &
                        pids[$!]=$!
                    done
                done

                # wait for all pids
                for pid in ${pids[*]}; do
                    wait $pid
                done
                unset pids
            done
        done
    done
}


PS3='Please enter your choice: '
options=("debug" "exp1" "exp1_plot" "run_exp1_addon" "exp2" "exp2_plot" "run_exp2_addon" "exp3" "exp4" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "exp1")
            run_exp1
            ;;

        "exp1_plot")
            exp1_plot
            ;;

        "run_exp1_addon")
            run_exp1_addon
            ;;
        
        "exp2")
            run_exp2
            ;;

        "exp2_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --attack mimic"
            python exp2.py $COMMON_OPTIONS --plot
            ;;

        "run_exp2_addon")
            run_exp2_addon
            ;;

        "exp3")
            run_exp3
            ;;

        "Quit")
            break
            ;;

        "debug")
            # python exp1.py --use-cuda --debug --identifier "exp1_debug" -n 10 -f 0 --attack NA --LT --noniid --agg rfa
            python exp2.py --use-cuda --identifier debug -n 25 -f 5 --attack mimic --agg cm --noniid --debug
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done

