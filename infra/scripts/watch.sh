#!/usr/bin/env bash

# An implementation of the Linux command `watch`

function main() {
    clear

    TIME=$(($1))  # change input to int

    while true
    do
        $2
        
        sleep $TIME
        clear
    done
}

main "$@"