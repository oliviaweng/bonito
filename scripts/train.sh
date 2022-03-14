CMD="bonito train ./testing/dynamic-skip-shorten --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=8 --teacher=./training/baseline -f --modifier=shorten --testing"

echo $CMD
$CMD