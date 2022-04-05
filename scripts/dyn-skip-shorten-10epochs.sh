CMD="bonito train ./training/dynamic-skip-shorten-pretrained-student-10epochs --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-dynamic-skip-shorten.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/skip-shorten-teacher --epochs=4 --modifier=shorten --modifier-freq=1 -f"

echo $CMD
$CMD