"""Vendored common helpers: logger, preprocessing, vec envs, toy envs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from pz_risk.common.envs import (
    BitFlippingEnv,
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
    SimpleMultiObsEnv,
)
from pz_risk.common.logger import (
    DEBUG,
    Figure,
    FormatUnsupportedError,
    Image,
    Logger,
    Video,
    configure,
)
from pz_risk.common.preprocessing import (
    check_for_nested_spaces,
    is_image_space,
    is_image_space_channels_first,
)
from pz_risk.common.running_mean_std import RunningMeanStd
from pz_risk.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from pz_risk.common.utils import set_random_seed
from pz_risk.common.vec_env import DummyVecEnv, is_vecenv_wrapped, unwrap_vec_normalize, unwrap_vec_wrapper
from pz_risk.common.vec_env.base_vec_env import tile_images
from pz_risk.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from pz_risk.common.vec_env.vec_normalize import VecNormalize


def test_running_mean_std():
    rms = RunningMeanStd(shape=(2,))
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rms.update(data)
    assert rms.mean.shape == (2,)
    assert np.allclose(rms.mean, data.mean(axis=0), atol=0.2)


def test_preprocessing_image_and_nested():
    image = spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    assert is_image_space_channels_first(image) is True
    assert is_image_space(image) is True
    hw = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    assert is_image_space_channels_first(hw) is False
    box = spaces.Box(-1, 1, (4,), dtype=np.float32)
    assert is_image_space(box) is False
    check_for_nested_spaces(box)
    nested = spaces.Dict({"a": spaces.Dict({"b": box})})
    with pytest.raises(NotImplementedError):
        check_for_nested_spaces(nested)


def test_set_random_seed():
    set_random_seed(123, using_cuda=False)
    a = np.random.rand()
    set_random_seed(123, using_cuda=False)
    b = np.random.rand()
    assert a == b


def test_logger_and_helpers(tmp_path):
    video = Video(torch.zeros(2, 3, 4, 4), fps=10)
    assert video.fps == 10
    fig = Figure(None, close=True)
    assert fig.close is True
    image = Image(np.zeros((4, 4, 3)), "HWC")
    assert image.dataformats == "HWC"
    with pytest.raises(FormatUnsupportedError):
        raise FormatUnsupportedError(["csv"], "video")
    with pytest.raises(FormatUnsupportedError):
        raise FormatUnsupportedError(["csv", "json"], "video")

    logger = configure(str(tmp_path / "log"), ["stdout", "csv", "log"])
    logger.record("test/reward", 1.5)
    logger.dump(step=1)
    logger.close()
    assert DEBUG == 10


def test_identity_and_bitflip_envs():
    env = IdentityEnv(dim=4, ep_length=3)
    obs = env.reset()
    nxt, rew, done, info = env.step(obs)
    assert rew in (0.0, 1.0)
    env.render()
    box = IdentityEnvBox(ep_length=2)
    box.reset()
    box.step(np.array([0.0], dtype=np.float32))
    IdentityEnvMultiDiscrete(dim=2, ep_length=2).reset()
    IdentityEnvMultiBinary(dim=3, ep_length=2).reset()
    fake = FakeImageEnv(screen_height=8, screen_width=8, n_channels=1)
    fake.reset()
    fake.step(0)
    fake.render()

    bits = BitFlippingEnv(n_bits=4)
    bits.seed(0)
    obs = bits.reset()
    assert "observation" in obs
    nxt, rew, done, info = bits.step(0)
    assert "is_success" in info
    bits.render(mode="human")
    arr = bits.render(mode="rgb_array")
    assert arr is not None
    bits.close()

    disc = BitFlippingEnv(n_bits=3, discrete_obs_space=True)
    dobs = disc.reset()
    assert isinstance(dobs["observation"], int)
    disc.step(0)

    img = BitFlippingEnv(n_bits=3, image_obs_space=True, channel_first=True)
    iobs = img.reset()
    assert iobs["observation"].ndim == 3
    img.step(1)

    cont = BitFlippingEnv(n_bits=3, continuous=True)
    cont.reset()
    cont.step(np.array([0.5, -0.2, 0.1], dtype=np.float32))


def test_simple_multi_obs_env():
    env = SimpleMultiObsEnv(random_start=False, discrete_actions=True)
    obs = env.reset()
    assert "vec" in obs and "img" in obs
    nxt, rew, done, info = env.step(2)
    env.render()
    cont = SimpleMultiObsEnv(random_start=True, discrete_actions=False, channel_last=False)
    cont.reset()
    cont.step(np.array([0.1, 0.2, 0.9, 0.0]))


def test_dummy_vec_env_and_normalize():
    from tests.conftest import BoxStubEnv

    def _factory():
        return BoxStubEnv()

    venv = DummyVecEnv([_factory, _factory])
    obs = venv.reset()
    assert obs.shape[0] == 2
    venv.step_async(np.array([0, 1]))
    obs, rews, dones, infos = venv.step_wait()
    assert len(rews) == 2
    venv.seed(0)
    images = venv.get_images()
    assert images is not None
    venv.close()

    assert unwrap_vec_normalize(venv) is None
    assert unwrap_vec_wrapper(venv, VecNormalize) is None
    assert is_vecenv_wrapped(venv, VecNormalize) is False


def test_obs_space_helpers():
    space = spaces.Box(-1, 1, (3,), dtype=np.float32)
    keys, shapes, dtypes = obs_space_info(space)
    assert keys == [None]
    assert shapes[None] == (3,)
    converted = dict_to_obs(space, {None: np.zeros((2, 3))})
    assert converted.shape == (2, 3)

    from collections import OrderedDict

    dspace = spaces.Dict(OrderedDict(a=spaces.Box(0, 1, (2,), dtype=np.float32)))
    # gymnasium Dict.spaces may be a plain dict; the helper requires OrderedDict
    if not isinstance(dspace.spaces, OrderedDict):
        dspace.spaces = OrderedDict(dspace.spaces)
    keys, shapes, dtypes = obs_space_info(dspace)
    copied = copy_obs_dict(OrderedDict(a=np.ones((2, 2), dtype=np.float32)))
    assert "a" in copied

    tspace = spaces.Tuple((spaces.Box(0, 1, (1,), dtype=np.float32), spaces.Box(0, 1, (2,), dtype=np.float32)))
    keys, shapes, dtypes = obs_space_info(tspace)
    tup = dict_to_obs(tspace, {0: np.zeros((1, 1)), 1: np.zeros((1, 2))})
    assert isinstance(tup, tuple)


def test_tile_images_and_rmsprop():
    imgs = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    tiled = tile_images(imgs)
    assert tiled.ndim == 3
    params = [torch.zeros(3, requires_grad=True)]
    opt = RMSpropTFLike(params, lr=1e-3)
    loss = (params[0] ** 2).sum()
    loss.backward()
    opt.step()
