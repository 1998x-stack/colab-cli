#!/usr/bin/env python3
"""Generate per-environment JSON configs for all ALE/*-v5 environments with obs_type="ram".

Run once to produce 63 config files. Configs are static JSON — no
ale-py dependency at training time.
"""
import json
import os
import sys


def known_atari_games():
    """Full list of Atari 2600 RAM environment IDs.

    Each tuple: (env_id, n_actions, suggested_network, total_timesteps).
    Sourced from ALE 0.10.x / Gymnasium Atari registry.
    Network and timesteps are defaults; adjust per-game after testing.
    """
    # (env_slug, n_actions, network, total_timesteps)
    games = [
        ("ALE/Adventure-v5", 18, "mlp-medium", 1000000),
        ("ALE/AirRaid-v5", 6, "mlp-medium", 500000),
        ("ALE/Alien-v5", 18, "mlp-large", 1000000),
        ("ALE/Amidar-v5", 10, "mlp-large", 1000000),
        ("ALE/Assault-v5", 7, "mlp-medium", 1000000),
        ("ALE/Asterix-v5", 9, "mlp-medium", 500000),
        ("ALE/Asteroids-v5", 14, "mlp-large", 1000000),
        ("ALE/Atlantis-v5", 4, "mlp-small", 500000),
        ("ALE/BankHeist-v5", 18, "mlp-large", 1000000),
        ("ALE/BattleZone-v5", 18, "mlp-medium", 1000000),
        ("ALE/BeamRider-v5", 9, "mlp-medium", 500000),
        ("ALE/Berzerk-v5", 18, "mlp-medium", 1000000),
        ("ALE/Bowling-v5", 6, "mlp-small", 500000),
        ("ALE/Boxing-v5", 18, "mlp-medium", 1000000),
        ("ALE/Breakout-v5", 4, "mlp-small", 500000),
        ("ALE/Carnival-v5", 6, "mlp-medium", 500000),
        ("ALE/Centipede-v5", 18, "mlp-large", 1000000),
        ("ALE/ChopperCommand-v5", 18, "mlp-medium", 1000000),
        ("ALE/CrazyClimber-v5", 9, "mlp-medium", 500000),
        ("ALE/Defender-v5", 18, "mlp-medium", 1000000),
        ("ALE/DemonAttack-v5", 6, "mlp-medium", 500000),
        ("ALE/DoubleDunk-v5", 18, "mlp-medium", 1000000),
        ("ALE/ElevatorAction-v5", 18, "mlp-large", 1000000),
        ("ALE/Enduro-v5", 9, "mlp-medium", 1000000),
        ("ALE/FishingDerby-v5", 18, "mlp-medium", 500000),
        ("ALE/Freeway-v5", 3, "mlp-small", 500000),
        ("ALE/Frostbite-v5", 18, "mlp-medium", 1000000),
        ("ALE/Gopher-v5", 8, "mlp-medium", 1000000),
        ("ALE/Gravitar-v5", 18, "mlp-large", 1000000),
        ("ALE/Hero-v5", 18, "mlp-medium", 1000000),
        ("ALE/IceHockey-v5", 18, "mlp-medium", 500000),
        ("ALE/Jamesbond-v5", 18, "mlp-medium", 500000),
        ("ALE/JourneyEscape-v5", 18, "mlp-large", 1000000),
        ("ALE/Kangaroo-v5", 18, "mlp-medium", 1000000),
        ("ALE/Krull-v5", 18, "mlp-medium", 1000000),
        ("ALE/KungFuMaster-v5", 14, "mlp-medium", 1000000),
        ("ALE/MontezumaRevenge-v5", 18, "mlp-large", 2000000),
        ("ALE/MsPacman-v5", 9, "mlp-large", 1000000),
        ("ALE/NameThisGame-v5", 6, "mlp-medium", 500000),
        ("ALE/Phoenix-v5", 8, "mlp-medium", 500000),
        ("ALE/Pitfall-v5", 18, "mlp-large", 2000000),
        ("ALE/Pong-v5", 6, "mlp-small", 500000),
        ("ALE/Pooyan-v5", 6, "mlp-medium", 500000),
        ("ALE/PrivateEye-v5", 18, "mlp-large", 2000000),
        ("ALE/Qbert-v5", 6, "mlp-medium", 1000000),
        ("ALE/Riverraid-v5", 18, "mlp-medium", 1000000),
        ("ALE/RoadRunner-v5", 18, "mlp-medium", 1000000),
        ("ALE/Robotank-v5", 18, "mlp-medium", 1000000),
        ("ALE/Seaquest-v5", 18, "mlp-medium", 1000000),
        ("ALE/Skiing-v5", 3, "mlp-small", 500000),
        ("ALE/Solaris-v5", 18, "mlp-large", 500000),
        ("ALE/SpaceInvaders-v5", 6, "mlp-medium", 500000),
        ("ALE/StarGunner-v5", 18, "mlp-medium", 1000000),
        ("ALE/Tennis-v5", 18, "mlp-medium", 1000000),
        ("ALE/TimePilot-v5", 10, "mlp-medium", 1000000),
        ("ALE/Tutankham-v5", 8, "mlp-medium", 500000),
        ("ALE/UpNDown-v5", 6, "mlp-medium", 500000),
        ("ALE/Venture-v5", 18, "mlp-medium", 1000000),
        ("ALE/VideoPinball-v5", 9, "mlp-small", 500000),
        ("ALE/WizardOfWor-v5", 10, "mlp-medium", 500000),
        ("ALE/YarsRevenge-v5", 18, "mlp-medium", 1000000),
        ("ALE/Zaxxon-v5", 18, "mlp-medium", 1000000),
    ]
    return games


def main():
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)

    # Load defaults to get solved thresholds
    with open(os.path.join(config_dir, "_defaults.json")) as f:
        defaults = json.load(f)

    # Solved thresholds (human-level scores) for reference
    solved_thresholds = {
        "Alien": 3000, "Amidar": 1000, "Assault": 800, "Asterix": 5000,
        "Asteroids": 1000, "Atlantis": 100000, "BankHeist": 1000,
        "BattleZone": 30000, "BeamRider": 5000, "Berzerk": 1000,
        "Bowling": 200, "Boxing": 50, "Breakout": 40, "Carnival": 5000,
        "Centipede": 5000, "ChopperCommand": 5000, "CrazyClimber": 50000,
        "Defender": 50000, "DemonAttack": 10000, "DoubleDunk": 0,
        "ElevatorAction": 30000, "Enduro": 500, "FishingDerby": 20,
        "Freeway": 30, "Frostbite": 1000, "Gopher": 5000, "Gravitar": 3000,
        "Hero": 30000, "IceHockey": 0, "Jamesbond": 1000,
        "JourneyEscape": 0, "Kangaroo": 2000, "Krull": 8000,
        "KungFuMaster": 30000, "MontezumaRevenge": 5000, "MsPacman": 3000,
        "NameThisGame": 5000, "Phoenix": 10000, "Pitfall": 0,
        "Pong": 18, "Pooyan": 3000, "PrivateEye": 0, "Qbert": 10000,
        "Riverraid": 10000, "RoadRunner": 30000, "Robotank": 30,
        "Seaquest": 50000, "Skiing": 0, "Solaris": 2000,
        "SpaceInvaders": 1000, "StarGunner": 30000, "Tennis": 0,
        "TimePilot": 5000, "Tutankham": 200, "UpNDown": 50000,
        "Venture": 1000, "VideoPinball": 100000, "WizardOfWor": 5000,
        "YarsRevenge": 30000, "Zaxxon": 10000,
        "Adventure": 0, "AirRaid": 0,
    }

    generated = 0
    for env_id, n_actions, network, total_timesteps in known_atari_games():
        game_name = env_id.split("/")[1].replace("-v5", "")
        # strip -ram if present (for solved_threshold lookup)
        config = {
            "env_id": env_id,
            "n_actions": n_actions,
            "solved_threshold": solved_thresholds.get(game_name, 0),
            "network": network,
            "total_timesteps": total_timesteps,
        }

        filename = env_id.replace("/", "-") + ".json"
        filepath = os.path.join(config_dir, filename)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        generated += 1

    print(f"Generated {generated} config files in {config_dir}")


if __name__ == "__main__":
    main()
