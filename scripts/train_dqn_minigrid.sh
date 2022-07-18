ENV="MiniGrid-$1-v0"
REPLAY="$2"
ALL_ARGS=${@:3}

echo "Env: $ENV"
echo "Replay: $REPLAY"
echo "All args: $ALL_ARGS"

python train.py \
    --algo dqn \
    --env $ENV \
    --replay-buffer $REPLAY \
    --track --wandb-project-name better-replay \
    $ALL_ARGS
