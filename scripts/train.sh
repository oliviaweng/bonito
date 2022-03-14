CMD="bonito train ./testing/dynamic-skip-remove --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=8 --teacher=./training/baseline -f --modifier=remove --testing"

echo $CMD
$CMD