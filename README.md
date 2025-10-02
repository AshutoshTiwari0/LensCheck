# 🧠 Cataract Detection using CNN 

This project is a **image-based catarcat classification app** that uses **Deep Learning** and **Streamlit** to detect signs of cataract in images.  


---

## 🚀 Features  
- Classifies image into **2 categories**:  
  - Cataract
  - Normal 
- Web app deployment support (via `app.py`, `Procfile`, `requirements.txt`, etc.)  

---

## 📊 Model Architecture and Performance (Key Metrics)  
<img width="569" height="314" alt="image" src="https://github.com/user-attachments/assets/85e80bf0-577e-4130-9f8a-20ebb1eed6fe" />

Number of epochs: 10 
Accuracy: 98.62%
Validation Accuracy: 94.21%

## 🛠️ Tech Stack  
- **Python**  
- **Tenorflow**  
- **Streamlit** (for deployment)  

---

## 📂 Project Structure  

```bash
├── app.py                       # Main app script  
├── cataract.ipynb   # Notebook with model training   
├── requirements.txt             # Dependencies  
├── Procfile                     # Deployment config  
├── runtime.txt                  # Python runtime version  
├── setup.sh                     # Setup script for deployment   

---
```
## 🚀 Getting Started  

### 1. Clone the repository  
```bash
git clone https://github.com/AshutoshTiwari0/LensCheck.git

```
2. Move to directory
```bash
cd LensCheck
```
3. Install dependencies
``` bash
pip install -r requirements.txt
```
4. Run the app
``` bash
python app.py
```

For deployment on platforms like Heroku, use the provided Procfile and setup.sh.

⚠️ Disclaimer

This project is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
