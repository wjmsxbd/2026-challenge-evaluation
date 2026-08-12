import copy

from tqdm import trange

import omnigibson as og


def _seed_environment(seed):
    """Seed all evaluator RNGs immediately before an environment-local reset operation."""
    from omnigibson.eval.utils.eval_utils import seed_everything

    return seed_everything(int(seed))


class VectorEnvironment:
    def __init__(self, num_envs, config, seeds=None):
        self.num_envs = num_envs
        if seeds is None:
            seeds = [None] * num_envs
        if len(seeds) != num_envs:
            raise ValueError(f"Expected {num_envs} environment seeds, got {len(seeds)}")
        self.seeds = [None if seed is None else int(seed) for seed in seeds]
        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = []
        for index in trange(num_envs, desc="Loading environments"):
            if self.seeds[index] is not None:
                _seed_environment(self.seeds[index])
            self.envs.append(og.Environment(configs=copy.deepcopy(config), in_vec_env=True))

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def _seed_for_env(self, index):
        seeds = getattr(self, "seeds", None)
        return None if seeds is None else seeds[index]

    def step(self, actions, env_indices=None):
        """Step selected environments in one shared simulator tick.

        ``env_indices`` controls action application, observation reads, and
        task bookkeeping. The simulator's global scene list intentionally
        stays unchanged so PhysX tensor/contact views retain stable scene
        indices when only a subset of vector slots remains active.
        """
        indices = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        if len(actions) != len(indices):
            raise ValueError(f"Expected {len(indices)} actions for env_indices={indices}, got {len(actions)}")

        observations, rewards, terminates, truncates, infos = [], [], [], [], []
        for idx, action in zip(indices, actions):
            self.envs[idx]._pre_step(action)
        # Evaluation requires one rendered 30 Hz frame after every action.
        with og.sim.render_on_step(True):
            og.sim.step()
        for idx, action in zip(indices, actions):
            obs, reward, terminated, truncated, info = self.envs[idx]._post_step(action)
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminated)
            truncates.append(truncated)
            infos.append(info)
        return observations, rewards, terminates, truncates, infos

    def reset(self, get_obs=True, env_indices=None, **kwargs):
        indices = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        if get_obs:
            observations, infos = [], []
            for idx in indices:
                seed = self._seed_for_env(idx)
                if seed is not None:
                    _seed_environment(seed)
                obs, info = self.envs[idx].reset(get_obs=get_obs, **kwargs)
                observations.append(obs)
                infos.append(info)
            return observations, infos
        else:
            for idx in indices:
                seed = self._seed_for_env(idx)
                if seed is not None:
                    _seed_environment(seed)
                self.envs[idx].reset(get_obs=get_obs, **kwargs)

    def _resolve_env_indices(self, env_indices):
        indices = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        if not indices:
            raise ValueError("Synchronized vector reset requires at least one environment")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate vector environment indices: {indices}")
        if any(idx < 0 or idx >= self.num_envs for idx in indices):
            raise IndexError(f"Vector environment indices out of range for {self.num_envs} environments: {indices}")
        return indices

    def reset_scenes_synchronized(self, env_indices=None):
        """Restore selected scenes and advance them with one shared physics step.

        Restoring a scene can synchronize object topology and internally advance the global simulator. Reload every
        selected scene's saved state after all restores so those topology-related steps cannot create slot-order drift,
        then issue the single physics step that a normal ``Scene.reset`` performs.
        """
        indices = self._resolve_env_indices(env_indices)
        for idx in indices:
            self.envs[idx].scene.reset(step_physics=False)
        for idx in indices:
            scene = self.envs[idx].scene
            scene.load_state(state=copy.deepcopy(scene._initial_file["state"]), serialized=False)
        og.sim.step_physics()

    def reset_synchronized(self, get_obs=True, env_indices=None, **kwargs):
        """Reset selected environments without advancing earlier vector slots more than later ones.

        This mirrors ``Environment.reset`` while sharing the scene physics step, normal simulator step, and render
        flushes across slots. It is intended for synchronized evaluation resets, when no unselected environment is
        concurrently rolling out.
        """
        indices = self._resolve_env_indices(env_indices)
        unsupported = [
            idx
            for idx in indices
            if not getattr(self.envs[idx].task, "supports_synchronized_scene_reset", False)
        ]
        if unsupported:
            raise TypeError(
                "Synchronized vector reset is only implemented for tasks that explicitly support skipping their "
                f"per-scene reset; unsupported slots: {unsupported}"
            )
        self.reset_scenes_synchronized(env_indices=indices)
        for idx in indices:
            env = self.envs[idx]
            seed = self._seed_for_env(idx)
            if seed is not None:
                _seed_environment(seed)
            env.task.reset(env, reset_scene=False)
            env._reset_variables()

        if not get_obs:
            return None

        with og.sim.render_on_step(True):
            og.sim.step()
        for _ in range(3):
            og.sim.render()

        observations, infos = [], []
        for idx in indices:
            obs, obs_info = self.envs[idx].get_obs()
            observations.append(obs)
            infos.append({"obs_info": obs_info})
        return observations, infos

    def close(self):
        for env in self.envs:
            env.close()

    def __len__(self):
        return self.num_envs
