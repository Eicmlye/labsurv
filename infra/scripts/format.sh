#!/usr/bin/env bash

## get OS: https://www.cnblogs.com/fnlingnzb-learner/p/10657285.html
## getting the operating system type
# function get_os() {
#     if [[ "$(uname)" == "Darwin" ]]
#     then
#         return 1
#     elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]
#     then
#         return 2
#     elif [[ "$(expr substr $(uname -s) 1 10)" =~ MINGW((32)|(64))_NT ]]
#     then
#         return 0
#     else
#         # Unknown OS
#         return 3
#     fi
# }

## format the input directory or file by black, isort and flake8
## only format python scripts
function format_file() {
    if [[ -d $1 ]]
    then
        local CUR_DIR=$1"/"
        local SUB_DIRS=$(ls $1)
        for DIR in ${SUB_DIRS[*]}
        do
            format_file ${CUR_DIR}${DIR}
        done
    elif [[ $1 =~ (\.pyi?|\.ipynb)$ ]]
    then
        echo -e "\n======== $1 ========"

        # use config pyproject.toml
        black $1
        # use config pyproject.toml
        isort $1

        if [[ $1 =~ \.py$ ]]
        then
            # use config .flake8
            flake8 $1
            local FLAKE_OUTPUT=$?

            # "" for correct format and "0" for excluded files
            if [[ ! ${FLAKE_OUTPUT} =~ ^0?$ ]]
            then
                exit 1
            fi
        fi
    fi
}

## get git user info to prevent unsafe changes
function get_git_user_name() {
    local USER_NAME=$(git config user.name)
    if [[ ! ${USER_NAME} == "" ]]
    then
        echo ${USER_NAME}
    else
        echo -e "\033[33m[ERROR]\033[0m No valid user found. Set user name first"\
        "by\ngit config user.name <user_name>" >&2
        exit 1
    fi
}
function get_git_user_email() {
    local USER_EMAIL=$(git config user.email)
    if [[ ! ${USER_EMAIL} == "" ]]
    then
        echo ${USER_EMAIL}
    else
        echo -e "\033[33m[ERROR]\033[0m No valid user found. Set user info first"\
        "by\ngit config user.email <user_email>" >&2
        exit 1
    fi
}

## get git change list, only list modified and new files
function get_git_status_change() {
    if [[ ! $(git status | grep "Untracked files:") == "" ]]
    then
        echo -e "\033[33m[WARN]\033[0m Untracked files must be added first to enable formatting." >&2
    fi
    
    local OUTPUT_LIST=$(git status | awk '
    BEGIN { flag = 0 }
    /^((Changes to be committed)|(Changes not staged for commit)):/ { flag = 1 }
    NF == 0 { flag = 0 }
    flag == 1 && $1 !~ /^\(|(deleted:)/ && $0 !~ /:$/ { print $(NF) }
    ')

    echo ${OUTPUT_LIST[*]}
}

## show usage info for --help flag
function show_usage() {
    echo -e "usage: sh infra/scripts/format.sh [<options>]\n"
    echo -e "\t\033[33mIt is strongly recommended to use --git\033[0m and by"\
    "using this, .gitignore will be valid for format check.\n"

    echo -e "\t--git\t\tOnly check modified files and new files according"\
    "to git status info. If not specified, ALL the files will be checked."

    echo -e "\t-h, --help\tShow this usage text and exit."

    echo -e "\t--ignore-platform\n\t\t\tForce running regardless of the OS platform."\
    "This may result in unexpected behaviours in non-Windows OS."
}

## main function
function main() {
    # HOME_DIR=$(cd $(dirname $0); pwd)
    CUR_WORKING_DIR=$(cd $(dirname ./); pwd)
    if [[ ! ${CUR_WORKING_DIR} =~ /labsurv$ ]]
    then
        echo -e "\033[31m[ERROR]\033[0m This script should be run at the root directory"\
        "of the project labsurv. Current working directory: ${CUR_WORKING_DIR}" >&2
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

    TARGET_DIR=$(ls ./)
    if [[ ${USE_GIT_STATUS} -eq 1 ]]
    then
        echo -e "\033[33mFound user [$(get_git_user_name)]\033[0m"
        echo -e "\033[33mFound email [$(get_git_user_email)]\033[0m"
        TARGET_DIR=$(get_git_status_change)
    else
        WAIT_TIME=10

        echo -e "\033[33m[WARN] ALL THE FILES will be formatted\033[0m in ${CUR_WORKING_DIR}."
        echo -e "\033[33m[WARN]\033[0m This script will abort automatically after ${WAIT_TIME} seconds."
        echo "Enter Y to continue. Any other entries will abort the script."

        if read -r -t ${WAIT_TIME} CONTINUE
        then
            if [[ ! ${CONTINUE} == "Y" ]]
            then
                echo "Format aborted. Use --git to only format files recorded by Git."
                exit 0
            fi
        else
            echo "Time out after ${WAIT_TIME} seconds waiting for response."\
            "Use --git to only format files recorded by Git."
            exit 1
        fi

        unset WAIT_TIME
        unset CONTINUE
    fi

    for DIR in ${TARGET_DIR[*]}
    do
        format_file ${DIR}
    done
}

main "$@"