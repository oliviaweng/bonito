CMD="bonito train ./training/dynamic-skip-remove --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=24 --teacher=./training/baseline --modifier=remove -f"

echo $CMD
$CMD
