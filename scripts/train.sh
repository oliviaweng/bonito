CMD="bonito train ./testing/static-skip-shorten --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-static-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=24 -f"

echo $CMD
$CMD