CMD="bonito train ./training/kd-skip-remove-pretrained-student --no-amp --config=./bonito/models/configs/dna_r9.4.1@v1-static-skip-remove.toml --directory ./bonito/data/dna_r9.4.1/ --batch=16 --teacher=./training/baseline -f"

echo $CMD
$CMD