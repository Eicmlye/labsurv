from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

agent = Agent(Net(make("CartPole-v1", env_num=9)))  # 初始化训练器
agent.train(total_time_steps=20000)
agent.save("./")
# 创建用于测试的环境，并设置环境并行数为9，设置渲染模式为group_human
env = make("CartPole-v1", env_num=9, render_mode="group_human")
agent.set_env(env)  # 训练好的智能体设置需要交互的环境
obs, info = env.reset()  # 环境进行初始化，得到初始的观测值和环境信息
while True:
    action, _ = agent.act(obs)  # 智能体根据环境观测输入预测下一个动作
    # 环境根据动作执行一步，得到下一个观测值、奖励、是否结束、环境信息
    obs, r, done, info = env.step(action)
    if any(done): break
env.close()  # 关闭测试环境