{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "38c0277e-37ea-4b8d-a7b9-1849844d6bea",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:432: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:432: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import pandas as pd\n",
    "\n",
    "# Load the Iris dataset\n",
    "iris = load_iris()\n",
    "X = iris.data\n",
    "y = iris.target\n",
    "\n",
    "# Train a RandomForest Classifier\n",
    "clf = RandomForestClassifier(random_state=42)  # Setting a random state for reproducibility\n",
    "clf.fit(X, y)\n",
    "\n",
    "# Define a function to take user input for the Iris flower features\n",
    "def user_input_features():\n",
    "    st.sidebar.header('Input Features')\n",
    "    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))\n",
    "    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))\n",
    "    petal_length = st.sidebar.slider('Petal length (cm)', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))\n",
    "    petal_width = st.sidebar.slider('Petal width (cm)', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))\n",
    "\n",
    "    # Creating a DataFrame for the features\n",
    "    data = {'sepal_length': sepal_length,\n",
    "            'sepal_width': sepal_width,\n",
    "            'petal_length': petal_length,\n",
    "            'petal_width': petal_width}\n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features\n",
    "\n",
    "# Get user input\n",
    "df = user_input_features()\n",
    "\n",
    "# Display the user input\n",
    "st.subheader('User Input Features')\n",
    "st.write(df)\n",
    "\n",
    "# Predict the class of the input features\n",
    "prediction = clf.predict(df)\n",
    "prediction_proba = clf.predict_proba(df)\n",
    "\n",
    "# Display the prediction and corresponding probability\n",
    "st.subheader('Prediction')\n",
    "st.write(f\"Predicted Iris species: **{iris.target_names[prediction][0]}**\")\n",
    "\n",
    "st.subheader('Prediction Probability')\n",
    "st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73d4c1fc-2a71-484f-8e52-af6cd7282e6b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
