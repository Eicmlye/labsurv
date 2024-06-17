#! /bin/bash
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
            SUB_DIRS=$(ls $1)
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
        fi
    fi
}

function get_gitignore() {
    OUTPUT_LIST=()

    if [[ $1 =~ ".gitignore" ]]
    then
        while read -r line
        do
            if [[ ${line} =~ ^[^\s#].* ]] && [[ ${#line} > 1 ]]
            then
                OUTPUT_LIST+=(${line%?})
            fi
        done < $1

        echo ${OUTPUT_LIST[*]}
    else
        echo "Only .gitignore file allowed."
        echo ${OUTPUT_LIST[*]}
    fi
}

IGNORE_LIST=$(get_gitignore .gitignore)
# HOME_DIR=$(cd $(dirname $0); pwd)
for DIR in $(ls ./)
do
    format_file ${DIR} "${IGNORE_LIST[*]}"
done