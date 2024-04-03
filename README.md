# KeenSight - GPU Finder

Welcome to KeenSight - GPU Finder! This application helps you discover the best cloud GPU deals from various providers. You can customize your search based on GPU count, memory, provider, instance type, and more.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/KeenSightStreamLit/gpu-finder.git
   ```

2. Navigate to the project directory:

   ```bash
   cd repo
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set up your configurations using the options in the sidebar:
   - **Providers:** Choose from available providers like AWS, Azure, GCP, etc.
   - **GPU:** Select GPU count, memory, and specific GPU names.
   - **Instance:** Specify spot type (on-demand, interruptable), CPU range, and memory range.

2. Explore the best GPU deals with the provided filters and view the results in a graphical and tabular format.

## Project Structure

- `main.py`: Main code for running the GPU Finder application.
- `logo.png`: Logo image used in the user interface.
- `requirements.txt`: Lists the required Python packages for the application.

## Features

- **Dynamic Data Retrieval:** Fetches real-time GPU offers from various providers.
- **Customizable Filters:** Allows users to refine search criteria for GPU deals.
- **Interactive Visualizations:** Presents GPU price comparison in an interactive chart.
- **Data Transparency:** Displays detailed information about each GPU offer.

## Configuration

- `st.set_page_config`: Configures the Streamlit page title and layout settings.
- `fetch_gpu_catalog`: Retrieves GPU catalog data from GPUHunt providers.
- `retrieve_gpu_offers`: Fetches GPU offers based on user-selected filters.
- Sidebar options for selecting providers, GPU parameters, and instance settings.

## Getting Started

To run the application:

```bash
python main.py
```

Open your browser and navigate to the provided URL to start exploring the best cloud GPU deals!

## Note

- The data is cached for a specific time interval to improve performance and reduce API calls.
- Ensure you have valid API keys and permissions for accessing GPU offers from different providers.

For any questions or issues, feel free to reach out to the project maintainer(s).

Happy GPU hunting! 🚀🔍