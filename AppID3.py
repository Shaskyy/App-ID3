import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import time  # Untuk simulasi loading

# Load kedua model
with open('ID3_Fish.sav', 'rb') as model_file:
    fish_model = pickle.load(model_file)

with open('ID3_Fruit.sav', 'rb') as model_file:
    fruit_model = pickle.load(model_file)

# Mapping hasil prediksi ke nama spesies untuk ikan dan buah
fish_species_map = {
    0: 'Anabas testudineus', 1: 'Coilia dussumieri', 2: 'Otolithoides biauritus',
    3: 'Otolithoides pama', 4: 'Pethia conchonius', 5: 'Polynemus paradiseus',
    6: 'Puntius lateristriga', 7: 'Setipinna taty', 8: 'Sillaginopsis panijus'
}
fruit_name_map = {0: 'grapefruit', 1: 'orange'}

# Judul aplikasi
st.title("Aplikasi Prediksi Nama Buah atau Spesies Ikan Menggunakan Algoritma ID3 (Iterative Dichotomizer 3)")

# Dropdown untuk memilih mode prediksi
prediction_mode = st.selectbox("Pilih jenis prediksi:", ("Prediksi Ikan", "Prediksi Buah"))

if prediction_mode == "Prediksi Ikan":
    st.subheader("Prediksi Spesies Ikan")
    st.write("Masukkan nilai fitur ikan di bawah ini untuk memprediksi spesies ikan.")
    
    # Input data untuk prediksi ikan
    length = st.number_input("Length")
    weight = st.number_input("Weight")
    w_l_ratio = st.number_input("Weight-Length Ratio")
    
    # Konversi input pengguna menjadi array numpy
    input_data = (length, weight, w_l_ratio)
    input_data_np = np.array(input_data).reshape(1, -1)
    
    # Tombol prediksi dengan efek loading
    if st.button("Prediksi Spesies Ikan"):
        with st.spinner("Sedang memprediksi..."):
            time.sleep(1)  # Simulasi loading
            prediction = fish_model.predict(input_data_np)
            species = fish_species_map.get(prediction[0], 'Spesies tidak diketahui')
            st.write(f"Prediksi Spesies Ikan: **{species}**")
    
    # Opsi untuk menampilkan pohon keputusan dan aturan
    if st.radio("Tampilkan Pohon Keputusan (Decision Tree)?", ("Tidak", "Ya")) == "Ya":
        with st.spinner("Sedang memprediksi..."):
            st.write("### Visualisasi Pohon Keputusan (Decision Tree) - Ikan")
            fig, ax = plt.subplots(figsize=(30, 15))
            plot_tree(
                fish_model, 
                feature_names=['length', 'weight', 'w_l_ratio'], 
                class_names=list(fish_species_map.values()), 
                filled=True, 
                fontsize=10,
                rounded=True,
                ax=ax
            )
            st.pyplot(fig)
    
    if st.radio("Tampilkan Aturan (Rules)?", ("Tidak", "Ya")) == "Ya":
        st.write("### Aturan (Rules) dari Pohon Keputusan - Ikan")
        tree_rules = export_text(fish_model, feature_names=['length', 'weight', 'w_l_ratio'])
        st.text(tree_rules)

elif prediction_mode == "Prediksi Buah":
    st.subheader("Prediksi Nama Buah")
    st.write("Masukkan nilai fitur buah di bawah ini untuk memprediksi nama buah.")
    
    # Input data untuk prediksi buah
    diameter = st.number_input("Diameter")
    weight = st.number_input("Weight")
    red = st.number_input("Red")
    green = st.number_input("Green")
    blue = st.number_input("Blue")
    
    # Konversi input pengguna menjadi array numpy
    input_data = (diameter, weight, red, green, blue)
    input_data_np = np.array(input_data).reshape(1, -1)
    
    # Tombol prediksi dengan efek loading
    if st.button("Prediksi Nama Buah"):
        with st.spinner("Sedang memprediksi..."):
            time.sleep(1)  # Simulasi loading
            prediction = fruit_model.predict(input_data_np)
            name = fruit_name_map.get(prediction[0], 'Nama buah tidak diketahui')
            st.write(f"Prediksi Nama Buah : **{name}**")
    
    # Opsi untuk menampilkan pohon keputusan dan aturan
    if st.radio("Tampilkan Pohon Keputusan (Decision Tree)?", ("Tidak", "Ya")) == "Ya":
        with st.spinner("Sedang memprediksi..."):
            st.write("### Visualisasi Pohon Keputusan (Decision Tree) - Buah")
            fig, ax = plt.subplots(figsize=(30, 15))
            plot_tree(
                fruit_model, 
                feature_names=['diameter', 'weight', 'red', 'green', 'blue'], 
                class_names=list(fruit_name_map.values()), 
                filled=True, 
                fontsize=10,
                rounded=True,
                ax=ax
            )
            st.pyplot(fig)
    
    if st.radio("Tampilkan Aturan (Rules)?", ("Tidak", "Ya")) == "Ya":
        st.write("### Aturan (Rules) dari Pohon Keputusan - Buah")
        tree_rules = export_text(fruit_model, feature_names=['diameter', 'weight', 'red', 'green', 'blue'])
        st.text(tree_rules)

# Menampilkan daftar kode untuk spesies ikan dan nama buah
st.write("### Daftar Kode dan Spesies/Nama")
if prediction_mode == "Prediksi Ikan":
    with st.expander("Klik untuk melihat daftar kode spesies ikan"):
        for code, species in fish_species_map.items():
            st.write(f"{code} : {species}")
elif prediction_mode == "Prediksi Buah":
    with st.expander("Klik untuk melihat daftar kode nama buah"):
        for code, name in fruit_name_map.items():
            st.write(f"{code} : {name}")


st.markdown("---")
st.markdown("**2213020152 | Nafika S.M | 3B | Machine Learning**")
