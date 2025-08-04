# 🍄 Mushroom Edibility Classifier

A machine learning project to predict whether a mushroom is **edible** or **poisonous** based on its physical features.

> "Know if your 🍄 is edible or poisonous before you take a bite 😅"

## 🔍 Demo

Check out the live demo here: [Mushroom Classifier on Streamlit](https://musroomclassifier.streamlit.app/)

---

## 🚀 Features

- Simple and interactive **Streamlit** web app
- Classifies mushrooms as **edible** or **poisonous**
- User selects features from dropdowns like:
  - Cap shape
  - Cap surface
  - Cap color
  - Odor
  - Gill size
  - ...and more
- Instant prediction result based on your selected inputs

---

## 🧠 ML Model

- **Model Type**: Decision Tree / Random Forest (depending on your model)
- **Framework**: Scikit-learn
- **Dataset**: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend/ML**: Python, Pandas, Scikit-learn
- **Deployment**: Streamlit Cloud

---

## 📁 Project Structure

mushroom-classifier/
│_
  ├── app.py # Main Streamlit app
  ├── mushroom_model.pkl # Trained ML model (Pickle file)
  ├── requirements.txt # Dependencies
  ├── README.md # This file
  └── ...

---

## ✅ How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

## 📦 Requirements

```bash
streamlit
pandas
scikit-learn
```

## 🙋‍♂️ Author
Made with ❤️ by Rudraksh Tripathi

## 📄 License
This project is open-source and available under the MIT License.

## 🌟 Show your support
If you liked this project, give it a ⭐ on GitHub or share it!






