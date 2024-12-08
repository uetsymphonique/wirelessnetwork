import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# PHẦN a: Điều chế BFSK với f1 = 100Hz, f2 = 200Hz
# ----------------------

# Tham số
A = 1  # Biên độ tín hiệu
f1_a = 100  # Tần số sóng mang 1 (Hz) cho phần a
f2_a = 200  # Tần số sóng mang 2 (Hz) cho phần a
bit_rate = 100  # Tốc độ truyền bit (bit/s)
fs = 10000  # Tần số lấy mẫu (Hz)

# Tạo dữ liệu 10 bit ngẫu nhiên
np.random.seed(42)  # Để đảm bảo kết quả lặp lại
data = np.random.randint(0, 2, 10)
print("Dữ liệu 10 bit ngẫu nhiên:", data)

# Thời gian của mỗi bit
T_bit = 1 / bit_rate  # Thời gian của mỗi bit (s)
t = np.linspace(0, len(data) * T_bit, int(len(data) * fs * T_bit), endpoint=False)

# Tạo tín hiệu BFSK phần a
s_t_a = np.zeros_like(t)
for i, bit in enumerate(data):
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    if bit == 1:
        s_t_a[start_idx:end_idx] = A * np.sin(2 * np.pi * f1_a * t[start_idx:end_idx])
    else:
        s_t_a[start_idx:end_idx] = A * np.sin(2 * np.pi * f2_a * t[start_idx:end_idx])

# Biểu diễn tín hiệu BFSK phần a theo thời gian
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tín hiệu BFSK theo thời gian (f1 = 100Hz, f2 = 200Hz)")
plt.plot(t, s_t_a)
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.grid(True)

# Phân tích phổ tần số phần a
fft_vals_a = np.fft.fft(s_t_a)
fft_freqs_a = np.fft.fftfreq(len(t), d=1 / fs)
fft_magnitude_a = np.abs(fft_vals_a) / len(t)
max_freq_a = 500  # Giới hạn tần số hiển thị (Hz)
idx_limit_a = np.where(fft_freqs_a > max_freq_a)[0][0]  # Chỉ số tương ứng với max_freq

# Biểu diễn phổ tần số BFSK phần a
plt.subplot(2, 1, 2)
plt.title("Phổ tần số của tín hiệu BFSK (f1 = 100Hz, f2 = 200Hz)")
plt.plot(fft_freqs_a[:idx_limit_a], fft_magnitude_a[:idx_limit_a])
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# PHẦN b: Điều chế BFSK với f1 = 50Hz, f2 = 100Hz
# ----------------------

# Tham số
f1_b = 50  # Tần số sóng mang 1 (Hz) cho phần b
f2_b = 100  # Tần số sóng mang 2 (Hz) cho phần b

# Tạo tín hiệu BFSK phần b
s_t_b = np.zeros_like(t)
for i, bit in enumerate(data):
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    if bit == 1:
        s_t_b[start_idx:end_idx] = A * np.sin(2 * np.pi * f1_b * t[start_idx:end_idx])
    else:
        s_t_b[start_idx:end_idx] = A * np.sin(2 * np.pi * f2_b * t[start_idx:end_idx])

# Biểu diễn tín hiệu BFSK phần b theo thời gian
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tín hiệu BFSK theo thời gian (f1 = 50Hz, f2 = 100Hz)")
plt.plot(t, s_t_b)
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.grid(True)

# Phân tích phổ tần số phần b
fft_vals_b = np.fft.fft(s_t_b)
fft_freqs_b = np.fft.fftfreq(len(t), d=1 / fs)
fft_magnitude_b = np.abs(fft_vals_b) / len(t)
max_freq_b = 500  # Giới hạn tần số hiển thị (Hz)
idx_limit_b = np.where(fft_freqs_b > max_freq_b)[0][0]  # Chỉ số tương ứng với max_freq

# Biểu diễn phổ tần số BFSK phần b
plt.subplot(2, 1, 2)
plt.title("Phổ tần số của tín hiệu BFSK (f1 = 50Hz, f2 = 100Hz)")
plt.plot(fft_freqs_b[:idx_limit_b], fft_magnitude_b[:idx_limit_b])
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# PHẦN c: Điều chế AM với tín hiệu BFSK (phần a)
# ----------------------

# Tham số sóng mang AM
C = 3          # Biên độ sóng mang
fc = 2000      # Tần số sóng mang (Hz)

# Tạo tín hiệu điều chế AM từ BFSK phần a
m_t_am = (C + s_t_a) * np.sin(2 * np.pi * fc * t)

# Biểu diễn tín hiệu AM và tín hiệu BFSK gốc để so sánh
plt.figure(figsize=(12, 8))

# Biểu đồ tín hiệu BFSK gốc
plt.subplot(2, 1, 1)
plt.title("Tín hiệu BFSK gốc (f1 = 100Hz, f2 = 200Hz)")
plt.plot(t, s_t_a, label="Tín hiệu BFSK gốc")
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.legend()
plt.grid(True)

# Biểu đồ tín hiệu AM
plt.subplot(2, 1, 2)
plt.title("Tín hiệu sau điều biến AM theo thời gian (dựa trên tín hiệu BFSK)")
plt.plot(t, m_t_am, label="Tín hiệu AM", color='orange')
plt.xlabel("Thời gian (s)")
plt.ylabel("m(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------
# PHẦN d: Mô phỏng tín hiệu AM tại bên nhận (có suy hao và nhiễu)
# ----------------------

# Tham số nhiễu và suy hao
lo = 0.5       # Hệ số suy hao
An = 0.2       # Cường độ nhiễu (AWGN)

# Tạo nhiễu trắng AWGN
noise = An * np.random.normal(0, 1, len(t))  # Nhiễu Gaussian

# Tín hiệu tại bên nhận
r_t = lo * m_t_am + noise

# Biểu diễn tín hiệu tại bên nhận
plt.figure(figsize=(12, 6))

# Biểu đồ tín hiệu tại bên nhận
plt.title("Tín hiệu tại bên nhận (có suy hao và nhiễu)")
plt.plot(t, r_t, label="Tín hiệu nhận", color='green')
plt.xlabel("Thời gian (s)")
plt.ylabel("r(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------
# PHẦN e: Giải điều biến AM và BFSK, xác định tỷ lệ lỗi bit
# ----------------------

# Giải điều chế AM
# Tách biên độ tín hiệu từ tín hiệu nhận được
am_demodulated = r_t * np.sin(2 * np.pi * fc * t)

# Lọc thông thấp (Low-pass filter) để lấy lại tín hiệu BFSK
fft_vals = np.fft.fft(am_demodulated)
fft_freqs = np.fft.fftfreq(len(t), d=1 / fs)

# Loại bỏ các tần số cao hơn BFSK (lọc thông thấp)
fft_vals[np.abs(fft_freqs) > 300] = 0  # Chỉ giữ tần số dưới 300Hz (phù hợp với BFSK)
filtered_signal = np.real(np.fft.ifft(fft_vals))

# Biểu diễn tín hiệu sau giải điều chế AM
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tín hiệu sau giải điều biến AM (BFSK)")
plt.plot(t, filtered_signal, label="Tín hiệu BFSK sau giải điều chế AM", color="blue")
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.legend()
plt.grid(True)

# ----------------------
# Giải điều chế BFSK
# ----------------------
# Tính năng lượng tại từng bit để giải điều chế BFSK
recovered_bits = []
for i in range(len(data)):
    # Lấy tín hiệu tương ứng với từng bit
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    segment = filtered_signal[start_idx:end_idx]

    # Tính tích phân tín hiệu tương ứng với f1 và f2
    energy_f1 = np.trapezoid(segment * np.sin(2 * np.pi * f1_a * t[start_idx:end_idx]), dx=1 / fs)
    energy_f2 = np.trapezoid(segment * np.sin(2 * np.pi * f2_a * t[start_idx:end_idx]), dx=1 / fs)

    # Quyết định bit dựa trên năng lượng
    if energy_f1 > energy_f2:
        recovered_bits.append(1)
    else:
        recovered_bits.append(0)

# Chuyển đổi mảng dữ liệu thu được
recovered_bits = np.array(recovered_bits)

# ----------------------
# Tính tỷ lệ lỗi bit (BER)
# ----------------------
errors = np.sum(data != recovered_bits)
ber = errors / len(data)

print(f"Số bit lỗi: {errors}")
print(f"Tỷ lệ lỗi bit (BER): {ber}")

# Biểu đồ dữ liệu gốc và dữ liệu khôi phục
plt.subplot(2, 1, 2)
plt.title("So sánh dữ liệu gốc và dữ liệu khôi phục")
plt.step(range(len(data)), data, label="Dữ liệu gốc", where="mid", color="blue")
plt.step(range(len(data)), recovered_bits, label="Dữ liệu khôi phục", where="mid", color="orange", linestyle="--")
plt.xlabel("Bit index")
plt.ylabel("Bit value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------
# PHẦN f: Quan hệ giữa SNR và BER đối với BFSK
# ----------------------

# Thay đổi giá trị biên độ nhiễu An để tính SNR và BER
noise_levels = np.linspace(0.001, 1, 100)  # Giá trị biên độ nhiễu từ nhỏ đến lớn
snr_values = []  # Danh sách lưu giá trị SNR
ber_values = []  # Danh sách lưu giá trị BER

for An in noise_levels:
    # Tạo nhiễu với biên độ An
    noise = An * np.random.normal(0, 1, len(t))

    # Tín hiệu tại bên nhận với nhiễu
    r_t = lo * m_t_am + noise

    # Giải điều chế AM và BFSK
    # Giải điều chế AM
    am_demodulated = r_t * np.sin(2 * np.pi * fc * t)
    fft_vals = np.fft.fft(am_demodulated)
    fft_vals[np.abs(fft_freqs) > 300] = 0
    filtered_signal = np.real(np.fft.ifft(fft_vals))

    # Giải điều chế BFSK
    recovered_bits = []
    for i in range(len(data)):
        start_idx = int(i * T_bit * fs)
        end_idx = int((i + 1) * T_bit * fs)
        segment = filtered_signal[start_idx:end_idx]
        energy_f1 = np.trapezoid(segment * np.sin(2 * np.pi * f1_a * t[start_idx:end_idx]), dx=1 / fs)
        energy_f2 = np.trapezoid(segment * np.sin(2 * np.pi * f2_a * t[start_idx:end_idx]), dx=1 / fs)
        recovered_bits.append(1 if energy_f1 > energy_f2 else 0)

    # Chuyển đổi dữ liệu khôi phục thành numpy array
    recovered_bits = np.array(recovered_bits)

    # Tính tỷ lệ lỗi bit (BER)
    errors = np.sum(data != recovered_bits)
    ber = errors / len(data)
    ber_values.append(ber)

    # Tính SNR
    signal_power = np.mean(m_t_am ** 2)  # Công suất tín hiệu
    noise_power = np.mean(noise ** 2)  # Công suất nhiễu
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    snr_values.append(snr_db)

# Biểu diễn quan hệ SNR và BER
plt.figure(figsize=(10, 6))
plt.plot(snr_values, ber_values, marker='o', linestyle='-', color='b', label='BFSK')
plt.title("Mối quan hệ giữa SNR và BER (BFSK)")
plt.xlabel("SNR (dB)")
plt.ylabel("Tỷ lệ lỗi bit (BER)")
plt.grid(True)
plt.legend()
plt.show()
