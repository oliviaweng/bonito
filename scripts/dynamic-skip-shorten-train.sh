CMD="bonito train ./training/dynamic-skip-shorten --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=24 --teacher=./training/baseline --modifier=shorten -f"

echo $CMD
$CMD