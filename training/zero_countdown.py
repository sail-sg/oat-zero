import launchpad as lp
from countdown import CountdownOracle
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program


class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: PPOArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.oracle = CountdownOracle()

        # Special treatment to sample from a base model - now only cover Qwen.
        self.sampling_params.stop = (
            ["</s>", "<|im_end|>", "<|endoftext|>", "\nUser:"]
            if "qwen" in args.pretrain.lower()
            else []
        )
        self.sampling_params.stop_token_ids = (
            [151645, 151643] if "qwen" in args.pretrain.lower() else []
        )

        self.eval_sampling_params.stop = (
            ["</s>", "<|im_end|>", "<|endoftext|>", "\nUser:"]
            if "qwen" in args.pretrain.lower()
            else []
        )
        self.eval_sampling_params.stop_token_ids = (
            [151645, 151643] if "qwen" in args.pretrain.lower() else []
        )


def run_ppo(args: PPOArgs):
    learner_cls = PPOLearner
    actor_cls = ZeroMathActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: PPOArgs = get_default_args(PPOArgs)

    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    run_ppo(args)
