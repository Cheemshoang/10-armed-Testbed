import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình trang
st.set_page_config(page_title="10-Armed Bandit Testbed", layout="wide")
st.title("🎰 Mô phỏng 10-Armed Testbed ($\epsilon$-Greedy)")
st.markdown("Thay đổi các thông số ở cột bên trái để xem cách thuật toán cân bằng giữa Khám phá và Khai thác.")

# --- SIDEBAR TƯƠNG TÁC ---
st.sidebar.header("⚙️ Cài đặt thông số")
k = st.sidebar.slider("Số lượng máy (k cánh tay)", min_value=2, max_value=20, value=10)
steps = st.sidebar.slider("Số bước thời gian (Steps)", min_value=100, max_value=5000, value=1000)
runs = st.sidebar.slider("Số lượt bài toán độc lập (Runs)", min_value=100, max_value=2000, value=2000, step=100)
epsilons_input = st.sidebar.text_input("Nhập các giá trị Epsilon (cách nhau bằng dấu phẩy)", "0.0, 0.01, 0.1")

# --- HÀM LÕI CHẠY THUẬT TOÁN (Tối ưu hóa bằng ma trận NumPy) ---
@st.cache_data # Cache dữ liệu để web không bị load lại từ đầu nếu không đổi cấu hình
def run_bandit_simulation(k, steps, runs, epsilons):
    results_reward = {}
    results_opt_action = {}
    
    # Khởi tạo giá trị thực q*(a) cho 'runs' bài toán, mỗi bài 'k' hành động
    q_true = np.random.normal(0, 1, (runs, k))
    true_opt_actions = np.argmax(q_true, axis=1) # Hành động tốt nhất thực sự
    
    for eps in epsilons:
        Q = np.zeros((runs, k)) # Giá trị ước lượng Q_t(a)
        N = np.zeros((runs, k)) # Số lần chọn N_t(a)
        
        rewards = np.zeros((runs, steps))
        opt_actions = np.zeros((runs, steps))
        
        for t in range(steps):
            # Sinh mảng xác suất ngẫu nhiên để quyết định Khám phá hay Khai thác
            rand_probs = np.random.rand(runs)
            explore_mask = rand_probs < eps
            
            # Khai thác (Greedy): Chọn hành động có Q lớn nhất
            action = np.argmax(Q, axis=1)
            
            # Khám phá (Explore): Ghi đè hành động ngẫu nhiên vào những vị trí explore_mask là True
            random_actions = np.random.randint(0, k, size=runs)
            action[explore_mask] = random_actions[explore_mask]
            
            # Lấy phần thưởng từ môi trường (có nhiễu phương sai = 1)
            reward = np.random.normal(q_true[np.arange(runs), action], 1)
            
            # Lưu lại thống kê
            rewards[:, t] = reward
            opt_actions[:, t] = (action == true_opt_actions)
            
            # Cập nhật giá trị Q và N (Sample-average)
            N[np.arange(runs), action] += 1
            Q[np.arange(runs), action] += (reward - Q[np.arange(runs), action]) / N[np.arange(runs), action]
            
        # Lấy trung bình qua tất cả các 'runs'
        results_reward[eps] = rewards.mean(axis=0)
        results_opt_action[eps] = opt_actions.mean(axis=0) * 100 # Chuyển sang phần trăm
        
    return results_reward, results_opt_action

# --- VẼ BIỂU ĐỒ ---
if st.sidebar.button("🚀 Chạy Mô Phỏng", type="primary"):
    try:
        epsilons = [float(e.strip()) for e in epsilons_input.split(",")]
    except:
        st.error("Vui lòng nhập Epsilon đúng định dạng số (vd: 0, 0.01, 0.1)")
        st.stop()

    with st.spinner('Đang tính toán trên hàng triệu điểm dữ liệu...'):
        rewards_dict, opt_actions_dict = run_bandit_simulation(k, steps, runs, epsilons)
    
    # Chia bố cục 2 cột cho 2 biểu đồ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trung bình phần thưởng (Average Reward)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for eps in epsilons:
            ax1.plot(rewards_dict[eps], label=f'$\epsilon = {eps}$')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Average Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.subheader("Tỷ lệ chọn hành động tối ưu (%)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for eps in epsilons:
            ax2.plot(opt_actions_dict[eps], label=f'$\epsilon = {eps}$')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('% Optimal Action')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
    st.success("✅ Đã hoàn thành mô phỏng!")