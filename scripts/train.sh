CMD="bonito train ./testing/dynamic-skip-remove-pretrained --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=8 --teacher=./training/baseline --modifier=remove -f --testing"

echo $CMD
$CMD