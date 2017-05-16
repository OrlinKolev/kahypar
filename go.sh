# a utility script for building, testing, etc.
# MUST SET KAHYPAR_TEST_DATA_DIR to parent directory of tests
#
# options:
#   -a --action : what to do (see below)
#       DEFAULT: 'run'
#
#   -r --refiners : space-separated list of refiners to use. Passed on to kahypar as --r-type, one 
#                   per invocation of kahypar
#       DEFAULT: empty; for every branch, set to refiner currently in development. 
#
#   -bt --build_type : DEBUG or RELEASE. DEBUG is actually RELWITHDBGINFO
#       DEFAULT: whatever was used on the last build
#
# possible actions:
#     run -- build and run once
#     build -- build only
#     commit -- prepare for commit, i.e. run with asserts on for all relevant refiners
#         (currently the default for the development branch and twoway_fm)
#     print -- print the arguments 'run' would use to with kahypar 
#     eval -- run several seeds, k-s, refiners and graphs

action='run'

# this is used on 'run' and 'commit'
# set to refiner currently in development
refiners='twoway_netstatus'

# this is used on 'eval' and 'commit'
baseline_refiners='twoway_fm'

build=true

while [[ $# -gt 1 ]]
do
    key="$1"

    case $key in
        -a|--action)
        action="$2"
        shift # past argument
        ;;
        -r|--refiners)
        refiners="$2"
        shift # past argument
        ;;
        -bt|--build_type)
        build_type="$2"
        shift # past argument
        ;;
        *)
                # unknown option
        ;;
    esac
    shift
done

if [ "$action" == 'eval' ]
then
    build_type='RELEASE'
fi

if [ "$action" == 'print' ]
then
    build=false
fi

if [ "$action" == 'commit' ]
then
    build_type='DEBUG'
    refiners="$refiners $baseline_refiners"
    action='run'
fi

if [ -n "$build_type" ]
then
    if [ "$build_type" == 'DEBUG' ]
    then
        build_type='RELWITHDBGINFO'
    fi

    cd build
    cmake .. -DCMAKE_BUILD_TYPE="$build_type"
    cd ..
fi

if [ "$build" = 'true' ]
then
    cd build
    make || exit 1
    cd ..
    if [ "$action" == 'build' ]
    then
        exit 0
    fi
fi

executable="./build/kahypar/application/KaHyPar"

test_ini="./config/km1_rb_dev17.ini"
eval_ini="./config/km1_rb_eval17.ini"

test_k="8"
eval_ks="2 4 8 16 32 64 128"

test_seed="-1"
eval_seeds="-1 42 666 664"

test_graph="ISPD98_ibm01.hgr"

eval_graphs=$(ls "$KAHYPAR_TEST_DATA_DIR")

args=( "--mode" "recursive" "--objective" "km1" "--epsilon" "0.03" )

if [ "$action" == 'eval' ]
then
    curr_seeds="$eval_seeds"
    curr_ks="$eval_ks"
    curr_graphs="$eval_graphs"
    refiners="$refiners $baseline_refiners"
    ini="$eval_ini"
    action='run'
else
    curr_seeds="$test_seed"
    curr_ks="$test_k"
    curr_graphs="$test_graph"
    ini="$test_ini"
fi

IFS=' '
for seed in $curr_seeds
do
for k in $curr_ks
do
for refiner in $refiners
do
IFS=$'\n'
for graph in $curr_graphs
do
    file="$KAHYPAR_TEST_DATA_DIR/$graph"
    if [ "$action" == 'run' ]
    then
        "$executable" "${args[@]}" --seed "$seed" -p "$ini" -k "$k" -h "$file" --r-type "$refiner" \
            || exit 1
    elif [ "$action" == 'print' ]
    then
        echo "$executable"
        echo "${args[@]}" --seed "$seed" -p "$ini" -k "$k" -h "$file" --r-type "$refiner"
    fi
done # graphs
IFS=' '
done # refiners
done # ks
done # seeds

rm -f "$KAHYPAR_TEST_DATA_DIR/*.KaHyPar"
