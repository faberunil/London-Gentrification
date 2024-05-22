# London Gentrification Data Visualization Dashboard

## Project Overview
This Streamlit application provides a comprehensive data dashboard for exploring and analysing gentrification across London boroughs. The dashboard visualizes data from 2010 to 2021, predicts gentrification trends up to 2030, and offers users the capability to input real-world data to assess gentrification levels in their areas.

## Environment
- **Python Version**: 3.12.1
- **Required Libraries**:
  - streamlit==1.33.0
  - matplotlib==3.8.3
  - seaborn==0.13.2
  - pandas==2.2.1
  - plotly==5.20.0
  - joblib==1.3.2
  - numpy==1.26.4
  - requests==2.31.0
  - scikit-learn==1.4.1.post1
  - tensorflow==2.16.1

## Installation

Clone the project repository:

```bash
git clone https://github.com/faberunil/London-Gentrification
cd London-Gentrification

pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```
## File Structure
/London-Gentrification
│
├── /Cleaning         - Scripts for data cleaning and preprocessing
├── /Data             - Raw and cleaned data files in CSV format
├── /EDA              - Exploratory data analysis scripts, df creation script
├── /Modeling         - Machine learning models and prediction scripts
├── /Models           - Trained model files and results datasets
├── streamlit_app.py  - Main Python script for the Streamlit application
├── requirements.txt  - Provides list of required libraries and their version
