# Sea Level Rise Prediction Project with Neural Networks

This project uses neural networks to analyze and predict sea level rise (GMSL) based on historical and future greenhouse gas emissions data (CO2, CH4, N2O). The data is extracted from IPCC AR6 reports.

## Project Structure

- **/data**: Contains raw data files in Excel format from the IPCC.  
- **/notebooks**: Contains Jupyter Notebooks for analysis, training, and model evaluation.  
- **/models**: Stores trained neural network models.  
- **/src**: Contains source code, such as the definition of neural network architectures.  
- **requirements.txt**: Lists the Python dependencies needed to run the project.  

## Environment Setup

To run the notebooks and scripts of this project, you need to set up a Python environment and install the required dependencies.

### 1. Clone the repository (if hosted on GitHub):

```bash
git clone <repository-url>
cd <repository-directory>
```
2. Create a virtual environment:
It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

```bash
python -m venv my-new-env
```
3. Activate the virtual environment:
On Windows:

```bash
.\my-new-env\Scripts\activate
On macOS/Linux:
```

```bash
source my-new-env/bin/activate
```

4. Install dependencies:
Use pip to install all the libraries listed in requirements.txt.


```bash
pip install -r requirements.txt
```

## How to Run the Notebooks
The notebooks are designed to be executed in a specific order to follow the workflow of the project, from data exploration to future scenario analysis.

Make sure you have the virtual environment activated and Jupyter Notebook or Jupyter Lab installed (pip install notebook).

Start the Jupyter server:
```bash
jupyter notebook
```
Then, open and run the notebooks in the following order from the notebooks/ folder:

00_data_exploration.ipynb: Loads initial data, cleans it, merges datasets, and performs exploratory analysis to understand the variables and their relationships.

01_model_training.ipynb: Defines and trains neural network models using the processed data. Trained models are saved in the /models folder.

02_model_analysis.ipynb: Loads saved models and evaluates their performance, comparing predictions against actual data.

03_future_scenario_analysis.ipynb: Uses trained models to predict sea level rise under different future emissions scenarios.
