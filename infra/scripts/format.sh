#!/usr/bin/env bash

# get OS: https://www.cnblogs.com/fnlingnzb-learner/p/10657285.html
function get_os() {
    if [[ "$(uname)" == "Darwin" ]]
    then
        return 1
    elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]
    then
        return 2
    elif [[ "$(expr substr $(uname -s) 1 10)" =~ MINGW((32)|(64))_NT ]]
    then
        return 0
    else
        # Unknown OS
        return 3
    fi
}

function format_file() {
    local IGNORE_LIST=$2
    local ignore_current_file=1

    if [[ -d $1 ]]
    then
        local CUR_DIR=$1"/"
        for IGNORE in ${IGNORE_LIST[*]}
        do
            if ([[ ${IGNORE} =~ ^/.* ]] && [[ "/"${CUR_DIR} == ${IGNORE} ]]) || [[ ${CUR_DIR} =~ ${IGNORE} ]]
            then
                ignore_current_file=0
                break
            fi
        done

        if [[ ${ignore_current_file} -eq 1 ]]
        then
            local SUB_DIRS=$(ls $1)
            for DIR in ${SUB_DIRS[*]}
            do
                format_file ${CUR_DIR}${DIR} "${IGNORE_LIST[*]}"
            done
        fi
    else
        for IGNORE in ${IGNORE_LIST[*]}
        do
            if ([[ ${IGNORE} =~ ^/.* ]] && [[ "/"$1 == ${IGNORE} ]]) || [[ $1 =~ ${IGNORE} ]]
            then
                ignore_current_file=0
                break
            fi
        done

        if [[ ${ignore_current_file} -eq 1 ]] && [[ ${DIR} =~ .*\.py$ ]]
        then
            echo $1
            black $1
            isort $1
            echo ""
        fi
    fi
}

# NOTE(eric): This function works for Windows
# Not sure if Linux generates this special char at the end of line
function get_gitignore() {
    local OUTPUT_LIST=()

    if [[ $1 =~ ".gitignore$" ]]
    then
        while read -r line
        do
            if [[ ${line} =~ ^[^\s#].* ]] && [[ ${#line} > 1 ]]
            then
                # delete the last special char at the end of line
                OUTPUT_LIST+=(${line%?})
            fi
        done < $1

        echo ${OUTPUT_LIST[*]}
    else
        echo "Only .gitignore files are allowed."
        echo ${OUTPUT_LIST[*]}
    fi
}

function get_git_status_change() {
    if [[ ! $(git status | grep "Untracked") == "" ]]
    then
        echo -e "\033[33m[WARN]\033[0m Untracked files must be added first to enable formatting." >&2
    fi
    
    local GIT_MODIFIED=$(git status | grep "modified:" | awk '{print $2}')
    for line in ${GIT_MODIFIED[*]}
    do
        GIT_MODIFIED+=(${line})
    done

    local GIT_NEW=$(git status | grep "new file:" | awk '{print $3}')
    for line in ${GIT_NEW[*]}
    do
        GIT_NEW+=(${line})
    done

    local OUTPUT_LIST=()
    OUTPUT_LIST+=${GIT_MODIFIED}
    OUTPUT_LIST+=${GIT_NEW}

    echo ${OUTPUT_LIST[*]}
}

function show_usage() {
    echo -e "usage: sh infra/scripts/format.sh [<options>]\n"

    echo -e "\t--git\t\tOnly check modified files and new files according to git status info."\
    "If not specified, files will be checked according to .gitignore."

    echo -e "\t-h, --help\tShow this usage text and exit."

    echo -e "\t--ignore-platform\n\t\t\tForce running regardless of the OS platform."\
    "This may result in unexpected behaviours in non-Windows OS."
}

# HOME_DIR=$(cd $(dirname $0); pwd)
CUR_WORKING_DIR=$(cd $(dirname ./); pwd)
if [[ ! ${CUR_WORKING_DIR} =~ labsurv/?$ ]]
then
    echo -e "\033[31m[ERROR]\033[0m This script should be run at the root directory of the project labsurv."\
    "Current working directory: ${CUR_WORKING_DIR}" >&2
    exit 1
fi

USE_GIT_STATUS=0
IGNORE_PLATFORM=0
HELP=0

# getopt usage: https://www.cnblogs.com/lxhbky/p/14189189.html
# getopt usage: https://blog.csdn.net/ARPOSPF/article/details/103381621
ARGS=`getopt -o h --longoptions git,ignore-platform,help -n "$0" -- "$@"`
if [ $? != 0 ]
then
    # found unknown options
    show_usage
    HELP=1
    exit 1
fi

eval set -- "${ARGS}"

while true
do
    case "$1" in
        --git) 
            USE_GIT_STATUS=1
            shift
            ;;
        -h|--help)
            HELP=1
            break
            ;;
        --ignore-platform)
            IGNORE_PLATFORM=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done


if [[ ${HELP} -eq 1 ]]
then
    show_usage
    exit 0
fi

get_os
OS=$?
if [[ ! ${OS} == 0 ]] && [[ ${IGNORE_PLATFORM} -eq 0 ]]
then
    echo -e "\033[33m[WARN]\033[0m Potential error may occur for git commands in non-Windows OS. "\
    "Use --ignore-platform to force running." >&2
    exit 1
fi

TARGET_DIR=$(ls ./)
if [[ ${USE_GIT_STATUS} -eq 1 ]]
then
    TARGET_DIR=$(get_git_status_change)
fi

IGNORE_LIST=$(get_gitignore .gitignore)

for DIR in ${TARGET_DIR[*]}
do
    format_file ${DIR} "${IGNORE_LIST[*]}"
done


# run command line in labsurv/