"""
Watch the trained PPO agent play PAC-man.

Usage:
    python3 ai_play.py                          # uses pacman_policy.onnx, level 1
    python3 ai_play.py --model my_model.onnx    # custom model path
    python3 ai_play.py --level 5                # start on level 5
    python3 ai_play.py --episodes 10            # play 10 episodes then exit

The game window opens normally — you can watch the AI play.
Close the window or press Ctrl+C to quit.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

_HERE      = os.path.dirname(os.path.abspath(__file__))
PACMAN_BIN = os.path.join(_HERE, "build", "pacman")


def load_model(model_path: str) -> ort.InferenceSession:
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found at {model_path}")
        print("       Run training first:  python3 train_ppo.py --envs 8")
        sys.exit(1)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"Loaded model: {model_path}")
    return sess


def get_action(sess: ort.InferenceSession, obs: np.ndarray) -> int:
    """Run inference and return the greedy action (argmax of logits)."""
    logits = sess.run(["action_logits"], {"state": obs[None].astype(np.float32)})[0]
    return int(np.argmax(logits, axis=-1)[0])


def play(model_path: str, level: int, max_episodes: int) -> None:
    sess = load_model(model_path)

    proc = subprocess.Popen(
        [PACMAN_BIN, "--rl", "--render", f"--level={level}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def send(msg: str) -> None:
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()

    def recv() -> dict:
        line = proc.stdout.readline()
        if not line:
            raise EOFError("Game process closed unexpectedly")
        return json.loads(line)

    # Read initial state sent by the game on startup
    data = recv()
    obs  = np.array(data["state"], dtype=np.float32)

    episode      = 1
    total_score  = 0
    episode_score = 0

    print(f"\nWatching AI play — Level {level}  (close window to quit)\n")

    try:
        while max_episodes == 0 or episode <= max_episodes:
            action = get_action(sess, obs)
            send(f'{{"action":{action}}}')
            data = recv()

            obs           = np.array(data["state"], dtype=np.float32)
            episode_score = data["score"]

            if data["done"]:
                lives = data["lives"]
                if lives > 0:
                    # Life lost but game continues — just keep playing, player will respawn
                    print(f"  Life lost — {lives} remaining  (score so far: {episode_score})")
                else:
                    # True game over (all lives gone)
                    total_score += episode_score
                    print(f"Episode {episode:>4}  score={episode_score:>6}  lives=0")
                    episode += 1

                    if max_episodes > 0 and episode > max_episodes:
                        break

                    # Reset for next episode
                    send('{"reset":true}')
                    data = recv()
                    obs  = np.array(data["state"], dtype=np.float32)

    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        proc.terminate()
        proc.wait()

    if episode > 1:
        print(f"\nPlayed {episode - 1} episodes  |  "
              f"avg score = {total_score / (episode - 1):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trained PPO agent play PAC-man")
    parser.add_argument("--model",    default="pacman_policy.onnx",
                        help="Path to ONNX model (default: pacman_policy.onnx)")
    parser.add_argument("--level",    type=int, default=1,
                        help="Starting level 1-10 (default: 1)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to play; 0 = play forever (default: 0)")
    args = parser.parse_args()

    if not os.path.isfile(PACMAN_BIN):
        print(f"ERROR: pacman binary not found at {PACMAN_BIN}")
        print("       Run:  cd build && cmake .. && make")
        sys.exit(1)

    play(
        model_path=args.model,
        level=args.level,
        max_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
