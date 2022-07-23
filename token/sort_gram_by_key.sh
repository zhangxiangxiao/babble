#!/bin/bash

# Sort space-seprated values by its first field.
#
# Usage: bash sort_data.sh [input_prefix] [output_prefix]

set -x;
set -e;

input_prefix=${1:-data/tinyshakespeare/train_gram_count};
output_prefix=${2:-data/tinyshakespeare/train_gram_count_sorted};

for input_file in ${input_prefix}*; do
    output_file=${output_prefix}${input_file#${input_prefix}}
    echo "sort -S 50% -k1,1 -o ${output_file} ${input_file};"
done;
