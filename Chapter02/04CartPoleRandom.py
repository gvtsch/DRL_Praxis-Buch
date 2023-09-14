import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        env.render()
        if done:
            break
    
    print(f"Episode nach {total_steps} Schritten beendet, Belohnung insgesamt {total_reward:.2f}")