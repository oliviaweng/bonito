CMD="bonito train ./training/dynamic-skip-remove-pretrained-student-10epochs --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/baseline --modifier=remove --modifier-freq=1 --epochs=4 --restore-optim -f"

echo $CMD
$CMD
