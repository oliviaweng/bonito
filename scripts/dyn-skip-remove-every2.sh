CMD="bonito train ./training/dynamic-skip-remove-pretrained-student-every2epochs --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/baseline --modifier=remove --modifier-freq=2 --epochs=4 -f"

echo $CMD
$CMD
