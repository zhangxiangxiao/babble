#!/bin/bash

# Sort space-seprated values by its first field.
#
# Additional arguments are passed to GNU sort. Some useful ones are "-S 50%" (using at
# maximum 50% of RAM) and "-T /mnt/scratch" (using /mnt/scratch as the temporary space),
# and --parallel=16 (using 16 threads).

set -x;
set -e;

input_prefix=${1:-data/tinyshakespeare/train_gram_count.};
output_prefix=${2:-data/tinyshakespeare/train_gram_count_sorted.};

for input_file in ${input_prefix}*; do
    output_file=${output_prefix}${input_file#${input_prefix}}
    sort -k1,1 -o ${output_file} ${input_file} "${@:3}";
done;
