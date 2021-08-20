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


function run_exp4 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 53 --noniid"
    for seed in 0 1 2
    do
        for s in 0 2 5
        do
            for m in 0 0.9
            do
                python exp4.py $COMMON_OPTIONS -f 5 --attack "IPM" --agg "cp" --bucketing $s --seed $seed --momentum $m &
                pids[$!]=$!
            done
        done

        for f in 1 6 12
        do
            for m in 0 0.9
            do
                python exp4.py $COMMON_OPTIONS -f $f --attack "IPM" --agg "cp" --bucketing 2 --seed $seed --momentum $m &
                pids[$!]=$!
            done
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}


function run_exp5 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
    for atk in "BF" "LF" "mimic" "IPM" "ALIE"
    do
        for m in 0 0.5 0.9 0.99
        do
            python exp5.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing 0 --seed 0 --momentum $m &
            pids[$!]=$!
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}


function run_exp6 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
    for atk in "BF" "LF" "mimic" "IPM" "ALIE"
    do
        for s in 0 2
        do
            python exp6.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing $s --seed 0 --momentum 0 &
            pids[$!]=$!

            for m in 0.5 0.9 0.99
            do
                python exp6.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing $s --seed 0 --momentum $m &
                pids[$!]=$!
                
                python exp6.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing $s --seed 0 --momentum $m --clip-scaling "linear" &
                pids[$!]=$!

                python exp6.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing $s --seed 0 --momentum $m --clip-scaling "sqrt" &
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

function run_exp6_addon {
    # TODO: delete after completion
    # Fix the bug for clip_scaling is None with momentum > 0
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
    for atk in "BF" "LF" "mimic" "IPM" "ALIE"
    do
        for s in 0 2
        do
            for m in 0.5 0.9 0.99
            do
                python exp6.py $COMMON_OPTIONS --attack $atk --agg "cp" --bucketing $s --seed 0 --momentum $m &
                pids[$!]=$!                
            done
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}

function run_exp7 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 20 -f 2 --noniid"
    for s in 0 2 3
    do
        python exp7.py $COMMON_OPTIONS --attack "LF" --agg "krum" --bucketing $s --seed 0 --momentum 0 &
        pids[$!]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function run_exp8 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
    for op in 1 2 4 8
    do
        for s in 0 2 3
        do
            for atk in "BF" "LF" "mimic" "IPM" "ALIE"
            do
                python exp8.py $COMMON_OPTIONS --attack $atk --agg "rfa" --bucketing $s --seed 0 --momentum 0 --op $op &
                pids[$!]=$!
            done
        done
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}


PS3='Please enter your choice: '
options=("debug" "exp1" "exp1_plot" "run_exp1_addon" "exp2" "exp2_plot" "run_exp2_addon" "exp3" "exp3_plot" "exp4" "exp4_plot" "exp5" "exp5_plot" "exp6" "exp6_plot" "run_exp6_addon" "run_exp7" "exp8" "Quit")
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

        "exp3_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
            python exp3.py $COMMON_OPTIONS --plot
            ;;

        "exp4")
            run_exp4
            ;;

        "exp4_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 53 --noniid"
            python exp4.py $COMMON_OPTIONS -f 5 --attack "IPM" --plot
            ;;

        "exp5")
            run_exp5
            ;;

        "exp5_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
            python exp5.py $COMMON_OPTIONS --attack "IPM" --agg "cp" --plot
            ;;

        "exp6")
            run_exp6
            ;;

        "exp6_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 25 -f 5 --noniid"
            python exp6.py $COMMON_OPTIONS --attack "IPM" --agg "cp" --plot
            ;;

        "run_exp6_addon")
            run_exp6_addon
            ;;

        "run_exp7")
            run_exp7
            ;;

        "exp8")
            run_exp8
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


