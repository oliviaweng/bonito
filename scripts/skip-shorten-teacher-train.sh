CMD="bonito train ./training/skip-shorten-teacher --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 -f"

echo $CMD
$CMD