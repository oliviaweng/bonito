CMD="bonito train ./training/dynamic-skip-shorten-pretrained-student --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/skip-shorten-teacher --epochs=3 --modifier=shorten --restore-optim -f"

echo $CMD
$CMD
