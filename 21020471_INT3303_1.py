import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ----------------------
# PHẦN 1a: Điều chế ASK
# ----------------------

# Tham số
A = 1  # Biên độ sóng cơ sở
f = 100  # Tần số sóng cơ sở (Hz)
bit_rate = 100  # Tốc độ truyền bit (bit/s)
fs = 10000  # Tần số lấy mẫu (Hz)
w0 = 0  # Pha ban đầu

# Tạo dữ liệu 10 bit ngẫu nhiên
np.random.seed(42)  # Để đảm bảo kết quả lặp lại
data = np.random.randint(0, 2, 10)
print("Dữ liệu 10 bit ngẫu nhiên:", data)

# Thời gian của mỗi bit
T_bit = 1 / bit_rate  # Thời gian của mỗi bit (s)
t = np.linspace(0, len(data) * T_bit, int(len(data) * fs * T_bit), endpoint=False)

# Tạo tín hiệu ASK
s_t = np.zeros_like(t)
for i, bit in enumerate(data):
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    n_t = bit
    s_t[start_idx:end_idx] = A * n_t * np.sin(2 * np.pi * f * t[start_idx:end_idx] + w0)

# Biểu diễn tín hiệu ASK theo thời gian
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tín hiệu ASK theo thời gian")
plt.plot(t, s_t)
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.grid(True)

# Phân tích phổ tần số (giới hạn tần số hiển thị)
fft_vals = np.fft.fft(s_t)
fft_freqs = np.fft.fftfreq(len(t), d=1 / fs)
fft_magnitude = np.abs(fft_vals) / len(t)
max_freq = 500  # Giới hạn tần số hiển thị (Hz)
idx_limit = np.where(fft_freqs > max_freq)[0][0]  # Chỉ số tương ứng với max_freq

# Biểu diễn phổ tần số với giới hạn
plt.subplot(2, 1, 2)
plt.title("Phổ tần số của tín hiệu ASK (Giới hạn từ 0 đến 500 Hz)")
plt.plot(fft_freqs[:idx_limit], fft_magnitude[:idx_limit])
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# PHẦN 1b: Điều chế AM
# ----------------------

# Tham số sóng mang AM
C = 3  # Biên độ sóng mang
fc = 2000  # Tần số sóng mang (Hz)

# Tạo tín hiệu AM
m_t = (C + s_t) * np.sin(2 * np.pi * fc * t + w0)

# Biểu diễn tín hiệu AM theo thời gian
plt.figure(figsize=(12, 6))  # Mở cửa sổ mới
plt.subplot(2, 1, 1)
plt.title("Tín hiệu AM theo thời gian")
plt.plot(t, m_t)
plt.xlabel("Thời gian (s)")
plt.ylabel("m(t)")
plt.grid(True)

# Phân tích phổ tần số của tín hiệu AM
fft_vals_am = np.fft.fft(m_t)
fft_freqs_am = np.fft.fftfreq(len(t), d=1 / fs)
fft_magnitude_am = np.abs(fft_vals_am) / len(t)
max_freq_am = 3000  # Giới hạn tần số hiển thị (Hz)
idx_limit_am = np.where(fft_freqs_am > max_freq_am)[0][0]  # Chỉ số tương ứng với max_freq_am

# Biểu diễn phổ tần số AM
plt.subplot(2, 1, 2)
plt.title("Phổ tần số của tín hiệu AM (Giới hạn từ 0 đến 3000 Hz)")
plt.plot(fft_freqs_am[:idx_limit_am], fft_magnitude_am[:idx_limit_am])
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# PHẦN 1c: Tín hiệu r(t) với nhiễu và suy hao
# ----------------------

# Tham số nhiễu và suy hao
lo = 0.5  # Hệ số suy hao tín hiệu
An = 0.2  # Cường độ nhiễu

# Tạo tín hiệu nhiễu AWGN
N_t = An * np.random.normal(0, 1, len(t))  # Nhiễu trung bình 0, phương sai 1

# Tạo tín hiệu r(t)
r_t = lo * (C + s_t) * np.sin(2 * np.pi * fc * t + w0) + N_t

# Biểu diễn tín hiệu r(t) theo thời gian
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tín hiệu r(t) theo thời gian (tín hiệu sau suy hao và nhiễu)")
plt.plot(t, r_t)
plt.xlabel("Thời gian (s)")
plt.ylabel("r(t)")
plt.grid(True)

# Phân tích phổ tần số của r(t)
fft_vals_r = np.fft.fft(r_t)
fft_freqs_r = np.fft.fftfreq(len(t), d=1 / fs)
fft_magnitude_r = np.abs(fft_vals_r) / len(t)
max_freq_r = 3000  # Giới hạn tần số hiển thị (Hz)
idx_limit_r = np.where(fft_freqs_r > max_freq_r)[0][0]  # Chỉ số tương ứng với max_freq_r

# Biểu diễn phổ tần số r(t)
plt.subplot(2, 1, 2)
plt.title("Phổ tần số của tín hiệu r(t) (Giới hạn từ 0 đến 3000 Hz)")
plt.plot(fft_freqs_r[:idx_limit_r], fft_magnitude_r[:idx_limit_r])
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# PHẦN 1d: Giải điều chế ASK và tính tỷ lệ lỗi bit
# ----------------------

# Chuẩn bị tín hiệu để giải điều chế (từ phần 1a)
demod_signal = s_t

# Tính tích phân mức năng lượng từng bit
recovered_bits = []
for i in range(len(data)):
    # Lấy đoạn tín hiệu tương ứng với một bit
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    signal_segment = demod_signal[start_idx:end_idx]

    # Tính mức năng lượng của đoạn tín hiệu
    energy = np.trapezoid(np.abs(signal_segment), dx=1 / fs)  # Lấy tích phân

    # Ngưỡng năng lượng để quyết định bit (bit 1 hoặc bit 0)
    threshold = (2 * A / (np.pi * f)) / 2  # Trung bình giá trị năng lượng lý thuyết của bit 1 và 0

    # So sánh mức năng lượng với ngưỡng để xác định bit
    recovered_bits.append(1 if energy > threshold else 0)

# So sánh dữ liệu gốc và dữ liệu khôi phục
data_recovered = np.array(recovered_bits)
errors = np.sum(data != data_recovered)  # Số bit lỗi
ber = errors / len(data)  # Tỷ lệ lỗi bit (BER)

# Hiển thị kết quả
print("Dữ liệu gốc:       ", data)
print("Dữ liệu khôi phục: ", data_recovered)
print(f"Số bit lỗi: {errors}")
print(f"Tỷ lệ lỗi bit (BER): {ber}")

# Biểu diễn tín hiệu đã giải điều chế và dữ liệu khôi phục
plt.figure(figsize=(12, 6))

# Biểu diễn tín hiệu ASK
plt.subplot(3, 1, 1)
plt.title("Tín hiệu ASK")
plt.plot(t, s_t)
plt.xlabel("Thời gian (s)")
plt.ylabel("s(t)")
plt.grid(True)

# Biểu diễn tín hiệu tích phân mức năng lượng từng bit
energy_levels = []
for i in range(len(data)):
    start_idx = int(i * T_bit * fs)
    end_idx = int((i + 1) * T_bit * fs)
    segment = demod_signal[start_idx:end_idx]
    energy_levels.append(np.trapezoid(np.abs(segment), dx=1 / fs))

plt.subplot(3, 1, 2)
plt.title("Năng lượng từng bit sau giải điều chế")
plt.bar(range(len(data)), energy_levels, color="blue")
plt.axhline(y=threshold, color="red", linestyle="--", label="Ngưỡng năng lượng")
plt.xlabel("Bit index")
plt.ylabel("Năng lượng")
plt.legend()
plt.grid(True)

# Biểu diễn dữ liệu gốc và dữ liệu khôi phục
plt.subplot(3, 1, 3)
plt.title("So sánh dữ liệu gốc và dữ liệu khôi phục")
plt.step(range(len(data)), data, label="Dữ liệu gốc", where="mid", color="blue")
plt.step(range(len(data)), data_recovered, label="Dữ liệu khôi phục", where="mid", color="orange", linestyle="--")
plt.xlabel("Bit index")
plt.ylabel("Bit value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ----------------------
# PHẦN 1e: Mối quan hệ giữa SNR và BER
# ----------------------

# Hàm tính tỷ lệ lỗi bit (BER)
def calculate_ber(data, signal, noise_level, lo, fs, T_bit, A):
    # Thêm nhiễu vào tín hiệu
    noise = noise_level * np.random.normal(0, 1, len(signal))
    r_t = lo * signal + noise

    # Nhân với sóng mang để giải điều chế
    r_t_mult = r_t * np.sin(2 * np.pi * f * t + w0)

    # Lọc thông thấp (Low-pass filter)
    fft_vals_mult = np.fft.fft(r_t_mult)
    fft_freqs_mult = np.fft.fftfreq(len(t), d=1 / fs)
    fft_vals_mult[np.abs(fft_freqs_mult) > f] = 0
    z_t = np.fft.ifft(fft_vals_mult).real

    # Khôi phục tín hiệu s(t)
    recovered_s_t = (2 * z_t / (lo * A)) - A

    # Giải điều chế ASK
    recovered_bits = []
    for i in range(len(data)):
        start_idx = int(i * T_bit * fs)
        end_idx = int((i + 1) * T_bit * fs)
        segment = recovered_s_t[start_idx:end_idx]
        energy = np.trapezoid(np.abs(segment), dx=1 / fs)  # Lấy tích phân năng lượng
        threshold = np.mean(energy_levels) # Ngưỡng năng lượng
        recovered_bits.append(1 if energy > threshold else 0)

    # So sánh dữ liệu gốc và dữ liệu khôi phục
    recovered_bits = np.array(recovered_bits)
    errors = np.sum(data != recovered_bits)
    ber = errors / len(data)
    return ber, noise


# Tham số tín hiệu và nhiễu
snr_values = []  # Giá trị SNR (dB)
ber_values = []  # Tỷ lệ lỗi bit (BER)
# noise_levels = np.linspace(0.05, 2, 20)  # Các mức nhiễu
noise_levels = np.linspace(0.001, 1, 100)  # Các mức nhiễu

snr_values = []  # Giá trị SNR (dB)
ber_values = []  # Tỷ lệ lỗi bit (BER)

for noise_level in noise_levels:
    ber, noise = calculate_ber(data, s_t, noise_level, lo, fs, T_bit, A)

    # Tính SNR
    signal_power = np.mean(s_t ** 2)  # Công suất tín hiệu
    noise_power = np.mean(noise ** 2)  # Công suất nhiễu
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)  # Chuyển sang đơn vị dB

    snr_values.append(snr_db)
    ber_values.append(ber)

# Biểu diễn mối quan hệ SNR và BER
plt.figure(figsize=(10, 6))
plt.plot(snr_values, ber_values, marker='o', linestyle='-', color='b')
plt.title("Mối quan hệ giữa SNR và BER")
plt.xlabel("SNR (dB)")
plt.ylabel("Tỷ lệ lỗi bit (BER)")
plt.grid(True)
plt.xlim(min(snr_values) - 5, max(snr_values) + 5)
plt.show()