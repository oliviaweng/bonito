CMD="bonito train ./testing/dynamic-skip-remove-pretrained-10epochs --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/baseline --modifier=remove --modifier-freq=1 --epochs=10 -f --testing"

echo $CMD
$CMD
