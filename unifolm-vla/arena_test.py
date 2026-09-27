# 麻雀虽小智能科技（武汉）有限公司
from lerobot.envs.factory import make_env

# =============================================================================
# 【中文说明】LeRobot Arena 环境接入最小示例
# =============================================================================
# 演示如何用 lerobot 的 make_env 加载 Isaac Lab Arena 的 Hub 环境，
# 并用**随机动作**跑一次闭环，用来确认"环境本身能构建、能 step"。
#
#   make_env("nvkartik/isaaclab-arena-envs", n_envs=4, trust_remote_code=True)
#     -> 返回 {任务名: [环境实例, ...]}；这里取 gr1_microwave 的第 0 个环境
#
# 【关键接口约定（ARENA 的 ArenaSimRobot 依赖同样的约定）】
#   env.reset()  -> (obs, info)
#   env.step(a)  -> (obs, reward, terminated, truncated, info)
#   terminated/truncated 是**批量**布尔数组（因为有 n_envs 维），
#   因此判断结束必须用 .any()，不能直接 if terminated。
#
# 【trust_remote_code=True 的含义】
#   Hub 环境需要下载并执行远端仓库里的代码。仅在信任该仓库时开启。
#
# 【与 ARENA 的关系】这是"环境接入参考"，不涉及策略推理。ARENA 把它抽象成
#   RobotInterface + ArenaSimRobot，从而与 VLA 客户端解耦。
# =============================================================================

envs_dict = make_env("nvkartik/isaaclab-arena-envs", n_envs=4, trust_remote_code=True)
env = envs_dict["gr1_microwave"][0]

obs, info = env.reset()
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        break
env.close()
