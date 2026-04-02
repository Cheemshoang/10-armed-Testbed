import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình trang
st.set_page_config(page_title="k-Armed Bandit Testbed", layout="wide")
st.title("Mô phỏng k-Armed Testbed")
st.markdown("Thay đổi các thông số ở cột bên trái để xem cách thuật toán cân bằng giữa Khám phá và Khai thác.")


st.sidebar.header("⚙️ Cài đặt thông số")
k = st.sidebar.slider("Số lượng k", min_value=2, max_value=20, value=10)
steps = st.sidebar.slider("Số time-steps", min_value=100, max_value=5000, value=1000)
runs = st.sidebar.slider("Số runs", min_value=100, max_value=2000, value=2000, step=100)
epsilons_input = st.sidebar.text_input("Nhập các giá trị Epsilon (cách nhau bằng dấu phẩy)", "0.0, 0.01, 0.1")


@st.cache_data 
def run_bandit_simulation(k, steps, runs, epsilons):
    results_reward = {}
    results_opt_action = {}
    #q_(a): gauss distribution for matrix runs x k
    q_true = np.random.normal(0, 1, (runs, k))
    # best action for each run
    true_opt_actions = np.argmax(q_true, axis=1) 
    
    for eps in epsilons:
        #init
        Q = np.zeros((runs, k))
        N = np.zeros((runs, k)) 
        rewards = np.zeros((runs, steps))
        opt_actions = np.zeros((runs, steps))
        
        for t in range(steps):
            rand_probs = np.random.rand(runs)
            explore_mask = rand_probs < eps
            #greedy
            action = np.argmax(Q, axis=1)

            # explore <-->explore_mask là True
            random_actions = np.random.randint(0, k, size=runs)
            # mask filter
            action[explore_mask] = random_actions[explore_mask]
            #reward gauss distribuion
            reward = np.random.normal(q_true[np.arange(runs), action], 1)
            #save
            rewards[:, t] = reward
            opt_actions[:, t] = (action == true_opt_actions)
            #sample avg
            N[np.arange(runs), action] += 1
            Q[np.arange(runs), action] += (reward - Q[np.arange(runs), action]) / N[np.arange(runs), action]
            
        # means-runs
        results_reward[eps] = rewards.mean(axis=0)
        results_opt_action[eps] = opt_actions.mean(axis=0) * 100 
        
    return results_reward, results_opt_action


if st.sidebar.button("Run model", type="primary"):
    try:
        epsilons = [float(e.strip()) for e in epsilons_input.split(",")]
    except:
        st.error("(vd: 0, 0.01, 0.1)")
        st.stop()

    with st.spinner('Thinking'):
        rewards_dict, opt_actions_dict = run_bandit_simulation(k, steps, runs, epsilons)
    
    # Chia bố cục 2 cột cho 2 biểu đồ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Reward")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for eps in epsilons:
            ax1.plot(rewards_dict[eps], label=f'$\epsilon = {eps}$')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Average Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.subheader("Optimal action (%)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for eps in epsilons:
            ax2.plot(opt_actions_dict[eps], label=f'$\epsilon = {eps}$')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('% Optimal Action')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
    st.success("✅Please I need this")