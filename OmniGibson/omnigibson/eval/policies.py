import logging
import numpy as np
import torch as th
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def set_action_dim(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        allow_reconnect: bool = False,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.last_action = None
        self.policy = None
        self._allow_reconnect = allow_reconnect
        if host is not None or port is not None:
            self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=allow_reconnect)

    def update_host(self, host: str, port: int) -> None:
        self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=self._allow_reconnect)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            return self.last_action
        response = self.infer(obs)
        self.last_action = th.from_numpy(np.asarray(response["action"]).copy()).to(th.float32).detach().cpu()
        return self.last_action

    def infer(self, obs: dict, *args, **kwargs) -> dict:
        """Return the complete websocket response, including optional action chunks."""
        if self.policy is None:
            raise RuntimeError("Websocket policy has not been connected to a server")
        return self.policy.infer(obs)

    def get_server_metadata(self) -> dict:
        if self.policy is None:
            return {}
        return self.policy.get_server_metadata() or {}

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None
