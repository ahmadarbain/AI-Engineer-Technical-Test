# ✨ AI Engineer - Technical Test Submission

Repository ini berisi penyelesaian untuk tes teknis posisi **AI Engineer**. Jawaban dikelompokkan ke dalam tiga folder terpisah sesuai dengan instruksi soal.

---

## 📁 Struktur Folder

├── customer-massage-classifier/ # Soal 1: Klasifikasi Pesan Pelanggan
├── ai-model-integration/ # Soal 2: Integrasi Model AI (Ollama)
└── fraud-detection-solution/ # Soal 3: Solusi Deteksi Kecurangan Reimbursement


---

## 🧠 Soal 1 - Project Python: Klasifikasi Pesan Pelanggan

### 🎯 Konteks
Perusahaan ISP menerima ribuan pesan harian dari pelanggan. Pesan ini perlu diklasifikasikan ke departemen yang sesuai secara otomatis menggunakan model ML.

### ✅ Ketentuan
- Model klasifikasi pesan ke 3 kategori:
  - `Information`
  - `Request`
  - `Problem`
- Bebas memilih metode/algoritma Machine Learning.
- Dataset dibagi: 80% training, 20% testing.
- Disertai evaluasi (Akurasi, Presisi, Recall).

### 📦 Isi Folder
- `dataset.csv`: Data pesan pelanggan yang sudah diberi label.
- `model_training.py`: Script pelatihan dan prediksi ML.
- `requirement.txt`: Daftar dependencies Python.
- `readme.txt`: Penjelasan metode, alur, instalasi & testing.

---

## 🤖 Soal 2 - Integrasi AI Model Ollama

### 🎯 Konteks
Mengintegrasikan AI chat model **Gemma 3B** dari [Ollama](https://ollama.com) ke dalam antarmuka pengguna berbasis Python.

### ✅ Ketentuan
- Install Ollama + model `gemma3:1b`.
- Buat antarmuka chat sederhana (input & output chat).
- Backend terhubung ke model Ollama lokal.
- Gunakan Python untuk implementasi.

### 📦 Isi Folder
- `chat_ui.py`: Aplikasi UI chat.
- `ollama_service.py`: Koneksi ke Ollama model.
- `requirement.txt`: Daftar dependencies.
- `readme.txt`: Instruksi instalasi, testing, dan deskripsi metode.

---

## 🔍 Soal 3 - Problem Solving: Deteksi Fraud Reimbursement

### 🎯 Konteks
Reimbursement fiktif atau dimanipulasi menyebabkan kerugian internal perusahaan.

### ✅ Ketentuan
- Usulan solusi AI/ML untuk mencegah/mendeteksi fraud.
- Penjelasan model AI/ML yang digunakan.
- Penjabaran data yang diperlukan.
- Desain infrastruktur minimal.
- Disimpan dalam file `.txt`.

### 📦 Isi Folder
- `fraud_solution.txt`: Penjelasan solusi lengkap.

---

## 🚀 Pengumpulan

Silakan clone atau akses repository ini secara publik:

📎 **Repo URL:** [https://github.com/ahmadarbain/All-In-One-AI-Engineer-Test](https://github.com/ahmadarbain/All-In-One-AI-Engineer-Test)

---

## 🙌 Catatan

- Semua kode telah diberi komentar penjelas.
- Project dipisahkan rapi per soal.
- Siap diinstal & diuji lokal dengan `requirements.txt`.

---
