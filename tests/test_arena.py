# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 框架单元测试。

覆盖范围与组织
------------------------------------------------------------------
本文件按"从底层契约到上层服务"的顺序组织，共 29 个测试，分五组：

1. **数据契约**（``types``）—— 观测往返、相机去重、动作块形状；
2. **归一化与适配器**（``config`` / ``adapter``）—— 归一化往返、图像缩放、
   状态归一化、关节限位裁剪；
3. **后端与服务器**（``backends`` / ``server``）—— mock 确定性、后端工厂、
   ``/act`` 响应、**真实探活**（``/health`` 200/503、上游 5xx 语义）；
4. **控制回路**（``client``）—— 用假机器人跑完一个 episode；
5. **Sim-to-Real**（``sim2real``）—— 课程单调性、L1 恒等、
   相机/动力学随机化、可复现性。

运行环境要求
------------------------------------------------------------------
**不需要** GPU、不需要 Isaac Lab、不需要真实策略服务器、不需要外网。

唯一例外是第 3 组的探活测试：它们会在 ``127.0.0.1`` 上临时起一个本地 HTTP
服务（``uvicorn`` 或标准库）来验证"健康 / 不健康 / 不可达"三种判定，
结束后立即关闭。因此本文件依赖 ``server`` 可选依赖（fastapi/uvicorn），
若未安装，这部分测试会失败；框架本身的其它部分不受影响。

设计取向
------------------------------------------------------------------
每个回归测试都对应一个**真实修复过的缺陷**，并在 docstring 中说明"旧的错误
行为是什么"。这样测试同时充当"缺陷档案"，避免同类问题再次引入。
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.adapter import ActionAdapter, EmbodimentAdapter, ObservationAdapter, resize_image
from arena.backends import MockPolicyBackend, OpenPIBackend, PolicyBackend, build_backend
from arena.client import ControlLoop, RobotInterface
from arena.config import (
    AdapterConfig,
    ArenaConfig,
    NormalizationType,
    ServerConfig,
    Sim2RealConfig,
    normalize,
    unnormalize,
)
from arena.server import PolicyServer
from arena.sim2real import (
    CURRICULUM,
    MAX_DYNAMICS_SCALE,
    MIN_DYNAMICS_SCALE,
    DomainRandomization,
    MAX_DYNAMICS_MAGNITUDE,
    Randomizer,
    Sim2RealCurriculum,
)
from arena.types import ActionChunk, Observation


# ==========================================================================
# 第 1 组：数据契约（arena.types）
# ==========================================================================


def test_observation_round_trip():
    """Observation 经过 to_dict/from_dict 应保持图像、状态、任务名一致。"""
    img = np.full((224, 224, 3), 7, dtype=np.uint8)
    obs = Observation(
        images={"head": img, "wrist": img},
        state=np.arange(7, dtype=np.float32),
        instruction="pick",
        task_name="task_a",
    )
    restored = Observation.from_dict(obs.to_dict())
    assert np.array_equal(restored.full_image, img)
    assert np.allclose(restored.state, obs.state)
    assert restored.task_name == "task_a"


def test_observation_round_trip_has_no_duplicate_camera():
    """往返后同一帧不应同时以 "full" 和 "full_image" 两个键出现。

    回归测试: 旧实现把 payload 里的 "full_image" 当成一台具名相机收进
    images, 再额外插入 images["full"], 导致同一帧被存了两份。
    """
    img = np.full((8, 8, 3), 7, dtype=np.uint8)
    wrist = np.full((8, 8, 3), 9, dtype=np.uint8)
    obs = Observation(images={"head": img, "wrist": wrist}, state=np.zeros(3, dtype=np.float32))
    restored = Observation.from_dict(obs.to_dict())

    # 第三人称视角在传输中规范化为单个 "full" 键, 不得再留下 "full_image"。
    assert sorted(restored.images) == ["full", "wrist"]
    assert len(restored.images) == len(set(restored.images))
    # 具名相机 (wrist) 必须按原键名还原, 且内容正确。
    assert np.array_equal(restored.images["wrist"], wrist)
    assert np.array_equal(restored.images["full"], img)
    # all_images 不能把同一帧算两次。
    assert len(restored.all_images) == 2


def test_observation_from_dict_legacy_alias_only():
    """只带别名键 (无 full_image) 的旧版 payload 也应能还原第三人称视角。"""
    img = np.full((8, 8, 3), 5, dtype=np.uint8)
    restored = Observation.from_dict(
        {"head": img, "state": np.zeros(3, dtype=np.float32), "instruction": "x"}
    )
    assert sorted(restored.images) == ["full"]
    assert np.array_equal(restored.full_image, img)


def test_mock_backend_stable_across_processes():
    """mock 后端必须在不同进程 (不同 PYTHONHASHSEED) 下产生相同动作。

    回归测试: 旧实现使用内置 hash(instruction), 受 PEP 456 哈希随机化影响,
    每次新进程都会给出不同的动作偏移, 使 dry run 不可复现。
    """
    script = (
        "import numpy as np;"
        "from arena.backends import MockPolicyBackend;"
        "from arena.config import ServerConfig;"
        "b=MockPolicyBackend(ServerConfig());"
        "a=b.infer([{'state':np.zeros(7,dtype=np.float32)}],'pick up the cube');"
        "print(repr(float(a[0,0])))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=str(Path(__file__).resolve().parents[1]),
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outputs) == 1, f"mock 后端跨进程不确定: {outputs}"


def test_action_chunk_shapes():
    """一维动作应自动升为 (1, D)。"""
    chunk = ActionChunk(actions=np.zeros(7))
    assert chunk.actions.shape == (1, 7)
    assert chunk.action_dim == 7


# ==========================================================================
# 第 2 组：归一化与具身适配器（arena.config / arena.adapter）
# ==========================================================================


def test_normalization_round_trips():
    """归一化 + 反归一化应还原原始数值。"""
    stats = {"q01": [0.0, -1.0], "q99": [1.0, 1.0]}
    x = np.array([[0.25, -0.5]], dtype=np.float32)
    restored = unnormalize(
        normalize(x, stats, NormalizationType.BOUNDS_Q99), stats, NormalizationType.BOUNDS_Q99
    )
    assert np.allclose(restored, x, atol=1e-5)


def test_resize_image():
    """图像应被缩放到目标边长且保持 uint8。"""
    out = resize_image(np.zeros((640, 480, 3), dtype=np.uint8), 224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_adapter_state_normalization():
    """观测适配器应对 proprio 应用归一化。"""
    stats = {"proprio": {"q01": [0.0], "q99": [1.0]}}
    adapter = ObservationAdapter(
        AdapterConfig(normalization_type=NormalizationType.BOUNDS_Q99), stats
    )
    assert adapter.encode_state(np.array([0.5], dtype=np.float32))[0] == pytest.approx(0.0, abs=1e-5)


def test_action_adapter_joint_limits():
    """动作应被裁剪到关节限位范围内。"""
    adapter = ActionAdapter(AdapterConfig(joint_lower=[-1.0, -1.0], joint_upper=[1.0, 1.0]))
    clipped = adapter.decode_action(np.array([5.0, -5.0], dtype=np.float32))
    assert clipped[0] == pytest.approx(1.0)
    assert clipped[1] == pytest.approx(-1.0)


# ==========================================================================
# 第 3 组：策略后端与服务器（arena.backends / arena.server）
# ==========================================================================


def test_mock_backend_deterministic():
    """mock 后端在同一输入下应产生确定性输出。"""
    backend = MockPolicyBackend(ServerConfig())
    obs = [{"state": np.zeros(7, dtype=np.float32)}]
    assert np.allclose(backend.infer(obs, "task"), backend.infer(obs, "task"))
    assert backend.infer(obs, "task").shape == (8, 7)


def test_build_backend_unknown():
    """未知后端名应抛出 ValueError。"""
    with pytest.raises(ValueError):
        build_backend(ServerConfig(backend="nope"))


def test_policy_server_act():
    """/act 应返回 (H, action_dim) 动作与延迟。"""
    server = PolicyServer(MockPolicyBackend(ServerConfig()))
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    response = server.act(
        {"observations": [{"state": np.zeros(7), "full_image": img, "instruction": "x"}]}
    )
    assert np.asarray(response["actions"]).shape[1] == 7
    assert "latency_s" in response


class _FakeRobot(RobotInterface):
    """测试用机器人：固定步数后判定成功。"""

    def __init__(self, horizon=4):
        self.horizon = horizon
        self.steps = 0

    def reset(self):
        self.steps = 0
        return self.get_observation()

    def get_observation(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        return {"images": {"head": img, "wrist": img}, "state": np.zeros(7, dtype=np.float32)}

    def execute(self, action):
        self.steps += 1

    def is_running(self):
        return self.steps < self.horizon

    def task_finished(self):
        return self.steps >= self.horizon


class _Client:
    """进程内策略客户端，直接调用后端。"""

    def __init__(self, backend):
        self.backend = backend

    def infer(self, observation, instruction=None):
        return ActionChunk(actions=self.backend.infer([observation.to_dict()], instruction or ""))


# ==========================================================================
# 第 4 组：控制回路（arena.client）
# ==========================================================================


def test_control_loop_completes():
    """控制回路应跑完一个 episode 并报告 success。"""
    adapter = EmbodimentAdapter(AdapterConfig(image_size=32))
    loop = ControlLoop(
        _FakeRobot(horizon=4), _Client(MockPolicyBackend(ServerConfig())), adapter, instruction="go"
    )
    assert loop.run()[0]["success"] is True


# ==========================================================================
# 第 5 组：Sim-to-Real 域随机化（arena.sim2real）
# ==========================================================================


def test_sim2real_curriculum_monotonic():
    """课程 1-4 级的扰动强度应单调不减。"""
    for level in range(1, 4):
        assert CURRICULUM[level].lighting <= CURRICULUM[level + 1].lighting
        assert CURRICULUM[level].observation_noise <= CURRICULUM[level + 1].observation_noise


def test_sim2real_advance_saturates():
    """advance() 应饱和于 Level 4。"""
    curriculum = Sim2RealCurriculum(Sim2RealConfig(level=1))
    for _ in range(10):
        curriculum.advance()
    assert curriculum.level == 4


def test_config_yaml_round_trip(tmp_path):
    """配置写入 YAML 再读回应保持一致。"""
    config = ArenaConfig()
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    assert ArenaConfig.from_yaml(path).server.port == config.server.port


# ---------------------------------------------------------------------------
# 后端探活 (health)
# ---------------------------------------------------------------------------


def test_mock_backend_health_is_true():
    """mock 后端不依赖外部资源, 应报告健康。"""
    backend = MockPolicyBackend(ServerConfig())
    assert backend.health() is True
    status = backend.status()
    assert status["healthy"] is True
    assert status["backend"] == "MockPolicyBackend"


def test_server_health_consults_backend_not_just_process():
    """服务器 /health 必须询问后端, 而不是恒返回 ok。"""

    class _Dead(PolicyBackend):
        def infer(self, observations, instruction):  # pragma: no cover
            raise AssertionError("infer should not be called by /health")

        def health(self) -> bool:
            return False

    server = PolicyServer(_Dead(ServerConfig()))
    payload = server.health()
    assert payload["healthy"] is False
    assert payload["status"] == "unavailable"
    assert payload["backend"] == "_Dead"


def test_server_health_survives_backend_exception():
    """后端探活抛异常时 /health 仍应返回结构化结果而非崩溃。"""

    class _Exploding(PolicyBackend):
        def infer(self, observations, instruction):  # pragma: no cover
            raise AssertionError("infer should not be called by /health")

        def health(self) -> bool:
            raise RuntimeError("boom")

    payload = PolicyServer(_Exploding(ServerConfig())).health()
    assert payload["healthy"] is False
    assert "RuntimeError" in payload["detail"]


@pytest.mark.parametrize("alive", [True, False])
def test_http_backend_health_probes_upstream(alive):
    """HTTP 后端应通过真实探测区分上游存活 / 不可达。"""
    import http.server
    import socketserver
    import threading

    backend_cls = OpenPIBackend
    if alive:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def do_HEAD(self):  # noqa: N802
                self.send_response(200)
                self.end_headers()

            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"actions":[[0.0]]}')

            def log_message(self, *args):  # noqa: D102
                pass

        httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            backend = backend_cls(ServerConfig(upstream_url=f"http://127.0.0.1:{port}/act"))
            assert backend.health() is True
            assert backend.status()["healthy"] is True
        finally:
            httpd.shutdown()
            httpd.server_close()
    else:
        # 绑定后立刻释放, 得到一个确定无人监听的端口。
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        backend = backend_cls(ServerConfig(upstream_url=f"http://127.0.0.1:{port}/act"))
        assert backend.health() is False
        assert backend.status()["healthy"] is False
        # 5xx (例如中间代理的 502) 不能当作存活。
        assert "502" in backend.status()["detail"] or "Error" in backend.status()["detail"]


def test_http_backend_reports_unhealthy_when_upstream_declares_503():
    """上游 /health 返回 503 时, 转发后端必须报告不健康 (而非仅"可达")。"""
    import threading
    import time

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/health")
    def _health():
        return JSONResponse(
            {"status": "unavailable", "healthy": False}, status_code=503
        )

    # 端口 0 => 由内核分配, 避免测试之间抢占同一个端口。
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn 未能在超时内启动"

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        backend = OpenPIBackend(ServerConfig(upstream_url=f"http://127.0.0.1:{port}/act"))
        assert backend.health() is False
        assert "unhealthy" in backend.status()["detail"]
    finally:
        server.should_exit = True
        for _ in range(100):
            if not server.started:
                break
            time.sleep(0.05)


def test_arena_act_returns_json_native_payload():
    """act() 必须返回可直接 json 序列化的内容 (不再依赖 json_numpy 的全局 patch)。"""
    import json

    server = PolicyServer(MockPolicyBackend(ServerConfig()))
    payload = server.act(
        {"observations": [{"state": np.zeros(7, dtype=np.float32), "instruction": "x"}]}
    )
    # 不先调用 json_numpy.patch() 也必须能序列化。
    encoded = json.dumps(payload)
    assert '"latency_s"' in encoded
    assert isinstance(payload["actions"], list)
    assert isinstance(payload["actions"][0], list)


# ---------------------------------------------------------------------------
# 官方 UnifoLM-VLA 部署服务器: /health 路由
# ---------------------------------------------------------------------------


def test_deploy_server_exposes_health_route():
    """部署服务器应暴露 /health, 并按健康与否返回 200/503。

    该模块在导入时会拉起 torch / tensorflow, 故用 AST 做静态断言, 保持测试轻量。
    """
    import ast

    source = (
        Path(__file__).resolve().parents[1]
        / "unifolm-vla"
        / "deployment"
        / "model_server"
        / "run_real_eval_server.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    server_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Unifolm_VLA_Server"
    )
    methods = {n.name for n in server_cls.body if isinstance(n, ast.FunctionDef)}
    assert "health" in methods, "Unifolm_VLA_Server 缺少 health() 方法"
    assert "run" in methods

    run_src = ast.unparse(
        next(n for n in server_cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    )
    assert "/health" in run_src, "run() 未注册 /health 路由"
    assert "app.get" in run_src, "run() 未用 GET 注册路由"
    assert "503" in run_src and "200" in run_src, "/health 未按健康状态区分 200/503"

    health_src = ast.unparse(
        next(n for n in server_cls.body if isinstance(n, ast.FunctionDef) and n.name == "health")
    )
    for key in ("model_loaded", "processor_loaded", "norm_stats_loaded"):
        assert key in health_src, f"health() 未检查 {key}"


# ---------------------------------------------------------------------------
# Sim-to-Real: camera + dynamics 随机化
# ---------------------------------------------------------------------------


def _sample_image(seed=0, size=32):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_sim2real_level1_is_identity():
    """Level 1 是基线仿真, camera/dynamics 必须完全不动。"""
    curriculum = Sim2RealCurriculum(Sim2RealConfig(level=1))
    image = _sample_image()
    state = np.linspace(-1, 1, 8).astype(np.float32)
    observation = curriculum.randomize_observation(
        {"images": {"head": image}, "state": state}
    )
    assert np.array_equal(observation["images"]["head"], image)
    assert np.array_equal(observation["state"], state)
    assert "camera_randomization" not in observation
    assert curriculum.randomize_dynamics().is_identity is True


def test_sim2real_camera_randomization_perturbs_image():
    """Level >= 2 时 camera 随机化应真正改变图像, 且保持形状与 dtype。"""
    curriculum = Sim2RealCurriculum(Sim2RealConfig(level=3))
    image = _sample_image()
    observation = curriculum.randomize_observation({"images": {"head": image}})
    perturbed = observation["images"]["head"]
    assert perturbed.shape == image.shape
    assert perturbed.dtype == np.uint8
    assert not np.array_equal(perturbed, image)
    assert set(observation["camera_randomization"]) == {"zoom", "shift_x", "shift_y", "noise_sigma"}


def test_sim2real_camera_applies_same_view_to_all_cameras():
    """所有相机必须共用同一个 camera 采样, 保持视角一致。"""
    curriculum = Sim2RealCurriculum(Sim2RealConfig(level=4))
    image = _sample_image()
    observation = curriculum.randomize_observation(
        {"images": {"head": image, "wrist": image}}
    )
    # 输入相同 + 同一个 camera 采样 => 输出应逐像素相同。
    assert np.array_equal(observation["images"]["head"], observation["images"]["wrist"])


def test_sim2real_dynamics_scales_within_bounds():
    """dynamics 采样应始终落在物理合理的范围内, 并随等级增强。"""
    for level in (2, 3, 4):
        curriculum = Sim2RealCurriculum(Sim2RealConfig(level=level))
        for _ in range(25):
            dynamics = curriculum.randomize_dynamics()
            for value in dynamics.describe().values():
                assert MIN_DYNAMICS_SCALE <= value <= MAX_DYNAMICS_SCALE
        assert dynamics.is_identity is False


def test_sim2real_dynamics_never_negative():
    """即使 profile 给出极端 magnitude, 也不能产生非物理 (<=0) 的缩放系数。"""
    randomizer = Randomizer(DomainRandomization(level=4, dynamics=99.0), seed=1)
    assert MAX_DYNAMICS_MAGNITUDE <= 0.5
    for _ in range(50):
        for value in randomizer.randomize_dynamics().describe().values():
            assert value >= MIN_DYNAMICS_SCALE > 0.0


def test_sim2real_randomization_is_reproducible():
    """同一 seed 的两个 randomizer 应产生完全相同的扰动序列。"""
    image = _sample_image()

    def run():
        curriculum = Sim2RealCurriculum(Sim2RealConfig(level=3, seed=7))
        return (
            curriculum.randomize_observation({"images": {"head": image}})["images"]["head"],
            curriculum.randomize_dynamics().describe(),
        )

    first_image, first_dynamics = run()
    second_image, second_dynamics = run()
    assert np.array_equal(first_image, second_image)
    assert first_dynamics == second_dynamics
