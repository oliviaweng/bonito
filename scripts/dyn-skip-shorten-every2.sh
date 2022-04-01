CMD="bonito train ./training/dynamic-skip-shorten-pretrained-student-every2epochs --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/skip-shorten-teacher --epochs=10 --modifier=shorten --modifier-freq=2 -f"

echo $CMD
$CMD