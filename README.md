# Smart Learning Analytics Platform

## Overview

The Smart Learning Analytics Platform is an open-source web application designed to visualize and analyze student engagement and performance data from digital learning platforms. Built using **React** and **Python**, this platform aims to provide educators and administrators with actionable insights to enhance learning outcomes.

## Features

* **Interactive Dashboards**: Visualize student engagement metrics, performance trends, and content utilization.
* **Data Import**: Upload CSV files containing student interaction and assessment data.
* **Performance Metrics**: Analyze key performance indicators such as average scores, time spent per activity, and content completion rates.
* **User-Friendly Interface**: Intuitive design for easy navigation and data interpretation.

## Technologies Used

* **Frontend**: JavaScript, HTML, CSS
* **Backend**: Python (Flask)
* **Data Processing**: Pandas, NumPy, Sckit-learn
* **Visualization**: Matplotlib, Plotly, Seaborn.
* **Deployement**: Streamlit.


### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sravan-7799/Smart-Learning-Analytics-Platform.git
   cd Smart-Learning-Analytics-Platform
   ```

2. Install frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

3. Install backend dependencies:

   ```bash
   cd ../backend
   pip install -r requirements.txt
   ```

4. Start the backend server:

   ```bash
   python app.py
   ```

5. Start the frontend development server:

   ```bash
   cd ../frontend
   npm start
   ```

The application will be accessible at `http://localhost:3000`.

## Usage

* **Upload Data**: Navigate to the "Upload" section to import your CSV files.
* **View Dashboards**: Access various dashboards from the navigation menu to explore different analytics.
* **Customize Views**: Use filters and settings to tailor the visualizations to your needs.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.

