# tactile_envs

Collection of robotics environments equipped with both vision and tactile sensing. The environments are built on top of Gymnasium.

Current available environments:
* `tactile_envs/Insertion-v0` (see [Insertion](scripts/test_env_insertion.py))
* `Door` (see [Door](scripts/test_env_door.py))
* `HandManipulateBlockRotateZFixed-v1` (see [HandManipulateBlockRotateZFixed](scripts/test_env_hand.py))
* `HandManipulateEggRotateFixed-v1` (see [HandManipulateEggRotateFixed](scripts/test_env_hand.py))
* `HandManipulatePenRotateFixed-v1` (see [HandManipulatePenRotateFixed](scripts/test_env_hand.py))

## Installation
To install `tactile_envs` in a fresh conda env:
```
conda create --name tactile_envs python=3.11
conda activate tactile_envs
pip install -r requirements.txt
```

Before running the environment code, make sure that you generate the tactile sensor collision meshes for the desired resolution. E.g., for 32x32 sensors:
```
python tactile_envs/assets/insertion/generate_pad_collisions.py --nx 32 --ny 32
```

### Test the available environment:
```
python scripts/test_env_insertion.py
```

## Citation
If you use these environments in your research, please cite the following paper:
```
@article{sferrazza2023power,
  title={The power of the senses: Generalizable manipulation from vision and touch through masked multimodal learning},
  author={Sferrazza, Carmelo and Seo, Younggyo and Liu, Hao and Lee, Youngwoon and Abbeel, Pieter},
  year={2023}
}
```

## Additional resources
Are you interested in more complex robot environments with high-dimensional tactile sensing? Check out [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)!

## References
This codebase contains some files adapted from other sources:
* Gymnasium-Robotics: https://github.com/Farama-Foundation/Gymnasium-Robotics
* robosuite: https://github.com/ARISE-Initiative/robosuite/tree/master
* vit-pytorch: https://github.com/lucidrains/vit-pytorch
* TactileSimulation: https://github.com/eanswer/TactileSimulation