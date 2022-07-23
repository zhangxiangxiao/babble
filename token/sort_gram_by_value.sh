#!/bin/bash

# Sort space-seprated values by its second field.
#
# Additional arguments are passed to GNU sort. Some useful ones are "-S 50%" (using at
# maximum 50% of RAM) and "-T /mnt/scratch" (using /mnt/scratch as the temporary space),
# and --parallel=16 (using 16 threads).

set -x;
set -e;

input=${1:-data/tinyshakespeare/train_gram_count_combined.txt};
output=${2:-data/tinyshakespeare/train_gram_count.txt};

sort -k2,2 -n -r -o ${output} ${input} "${@:3}";
