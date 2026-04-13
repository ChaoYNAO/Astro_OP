# Astro_OP
A simple tool for Astronmical Observation Planning 


# Obs_plan_v1.ipynb - v1.version
This is the first version of the sigle-objective observation plan.

The output example diagram is as shown in the figure,

<img width="1484" height="918" alt="image" src="https://github.com/user-attachments/assets/3b7891c5-53e0-4ed6-bb14-1a3468121e66" />


# Obs_plan_v2.ipynb - v2.version
In this version, we can formulate muti-objective observation plans.

In a recent update, the issue of time zones at different observation locations was taken into account, and log files for output were provided.

1. Before running the script, ensure you have the required Python libraries installed. You can install them via pip:

pip install numpy matplotlib astropy astroplan

2. Configuration & Setup

To generate your observation plan, you only need to modify the parameters in the "Run Test Area" at the bottom of the script.

site_key = 'TNO'  # e.g., 'GAO' for Lijiang, 'TNO' for Thai National Obs.
my_location = OBSERVATORIES[site_key]
my_date = "2026-01-01"
my_targets = [
    # Example of a variable star (Phases will be calculated)
    {'name': 'Algol', 'ra': '03:08:10.1', 'dec': '+40:57:20.3', 'period': 2.867, 't0': 2440953},
    
    # Example of a standard star (No phase calculation)
    {'name': 'Sirius', 'ra': '06:45:08', 'dec': '-16:42:58', 'period': 0, 't0': 0},
]


3. Tips for Advanced Use

Altitude Limit: You can change the red warning line on the dashboard by modifying the alt_limit parameter in plot_final_smart_phase_dashboard(..., alt_limit=75).

Max Display Curves: To prevent the line chart from becoming too cluttered, the tool defaults to plotting curves for only the first 5 targets. This can be adjusted using the max_display parameter.

The output example diagram is as shown in the figure,

<img width="2066" height="2578" alt="image" src="https://github.com/user-attachments/assets/fe35a7ba-cc17-485a-9474-e04a2d488502" />


# Acknowledgement

We express our gratitude to the iris robot developed by the Astronomy Society of Nanjing University for the inspiration it has provided.
ref:https://meteorcollector.github.io/2024/02/iris-description/#%E4%BB%8B%E7%BB%8D
