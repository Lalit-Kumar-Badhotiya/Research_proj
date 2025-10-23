
# Anomaly Detection Project

This project contains two anomaly detection models: a generalized model and a specific model.

## Local Setup

To set up and run this project locally, follow these steps:

### 1. Prerequisites

-   Python 3.x installed.
-   Git installed.

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

```bash
cd Genarlized Anomaly Model
```
OR
```bash
cd Specific Anomaly Model
```
Then 
```bash
mkdir data/train/good
mkdir data/test/all
```
In the good folder add the training data
<br>
Int the all folder add the testing data (pretty self-explanatory)


### 5. Train the Model

```bash
python train.py
```
This will give a padim_model.pth file and this is your trained model

### 6. Running the Applications

This project contains two separate web applications.

#### Generalized Anomaly Model

To run the web application for the generalized anomaly model, execute the following command:

```bash
python "Generalized Anomaly Model/app.py"
```

The application will be available at `http://127.0.0.1:5000`.

#### Specific Anomaly Model

To run the web application for the specific anomaly model, execute the following command:

```bash
python "Specific Anomaly Model/app.py"
```

The application will be available at `http://127.0.0.1:5000`.

**Note:** Both applications run on the same port. You can only run one at a time, or you can modify the `app.py` file to run on different ports.
