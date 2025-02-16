# Streamlit Normal Distribution Analysis

This project is a Streamlit web application that allows users to analyze sample distributions from a normal population. Users can input various sample sizes, the number of samples to take, and the desired confidence level. The application visualizes the results through smoothed density plots and displays sample means with confidence intervals.

## Project Structure

```
streamlit-app
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   └── utils
│       └── analysis.py # Contains functions for sampling and analysis
├── requirements.txt    # Lists project dependencies
└── README.md           # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit application, execute the following command in your terminal:
```
streamlit run src/app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Usage

1. Enter up to 4 sample sizes separated by commas (e.g., 500,50,10).
2. Specify the number of samples to be taken.
3. Input the desired confidence level (e.g., 0.95).
4. Click the button to visualize the results.

The application will display:
- Smoothed density plots for the sample distributions compared to the population density.
- Sample means with confidence intervals, indicating whether the population mean is included in the intervals.

## Dependencies

The project requires the following Python packages:
- Streamlit
- NumPy
- Matplotlib
- SciPy
- Seaborn

Make sure to install these packages using the provided `requirements.txt` file.

## License

This project is licensed under the MIT License.