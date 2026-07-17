#!/bin/bash

if ! OPTIONS=$(getopt \
	-o "" \
	-l "env:,h:,gamma:,bs:,steps:,beta:,args:,micro_bs:" \
	-- "$@"); then
	exit 1
fi

eval set -- "$OPTIONS"

while [ $# -gt 0 ]; do
	case $1 in
	--env)
		USER_ENV=${2}
		shift
		;;
	--h)
		H=${2}
		shift
		;;
	--gamma)
		GAMMA=${2}
		shift
		;;
	--bs)
		BS=${2}
		shift
		;;
	--steps)
		STEPS=${2}
		shift
		;;
	--beta)
		BETA=${2}
		shift
		;;
	--args)
		ARGS=${2}
		shift
		;;
	--micro_bs)
		MICRO_BS=${2}
		shift
		;;
	--)
		shift
		break
		;;
	-*)
		echo "$0: error - unrecognized option $1" 1>&2
		exit 1
		;;
	*) break ;;
	esac
	shift
done

if [ -z "$SYSTEM" ]; then
	SYSTEM=erm_occupancy_mcts
fi

if [ -z "$USER_ENV" ]; then
	USER_ENV=jumanji/maximum_state_entropy_exploration
fi

if [ -z "$H" ]; then
	H=500
fi

if [ -z "$GAMMA" ]; then
	GAMMA=0.99
fi

if [ -z "$BS" ]; then
	BS=64
fi

if [ -z "$STEPS" ]; then
	STEPS=256
fi

if [ -z "$BETA" ]; then
	BETA=0.01
fi

if [ -z "$MICRO_BS" ]; then
	MICRO_BS=4
fi

UNIQUE_TOKEN=$(date +"%Y%m%d%H%M%S")
ALL_SEEDS=()

for i in $(seq 1 $BS); do
	SEED=$((((RANDOM << 30) | (RANDOM << 15) | RANDOM) & 0x7fffffff))
	ALL_SEEDS+=($SEED)
done

JOB_COUNT=0

for IDX in $(seq 1 "$BS"); do
	CURR_SEED=${ALL_SEEDS[$((IDX - 1))]}

	{
		(
			export HYDRA_FULL_ERROR=1
			python src/risk_aware_gumdp/systems/erm_occupancy_mcts.py \
				env=$USER_ENV \
				arch.seed=${CURR_SEED} \
				system.h=$H \
				system.gamma=$GAMMA \
				system.expansion_steps=$STEPS \
				system.beta=$BETA \
				logger.kwargs.project=gumdp \
				logger.kwargs.name="${SYSTEM}/${H}/${GAMMA}/${USER_ENV}" \
				logger.kwargs.unique_token=$(uuidgen) \
				$ARGS
		)
		sleep 1
	} &

	((JOB_COUNT++))

	if ((JOB_COUNT >= MICRO_BS)); then
		wait -n
		((JOB_COUNT--))
	fi
done

wait $(jobs -p)
exit
