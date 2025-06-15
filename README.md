# Kiln-Simulation-App

**Overview**
This Kiln Simulation App uses machine learning (Random Forest Regressor) to model and predict kiln behavior. It takes key input features such as fuel rate, primary air, secondary air, and chemical composition parameters including CaO, SiO₂, Al₂O₃, and Fe₂O₃. Based on these inputs, the model accurately predicts two critical outputs: T_gas and T_solid. The app visualizes these predictions in real-time time-series graphs using Matplotlib, allowing users to observe and analyze the temperature progression throughout the simulation.

**Key Features**
• ML-Powered Predictions – Predicts T_gas and T_solid using a trained Random Forest Regressor model
• Flexible Input Controls – Adjust parameters like fuel rate, primary air, secondary air, CaO, SiO₂, Al₂O₃, and Fe₂O₃
• Real-Time Time-Series Graphs – Visualize output temperatures dynamically with Matplotlib
• Save & Load Profiles – Store your customized input configurations for future simulations
• Export Results – Download output data and charts for deeper analysis
• User-Friendly Interface – Built with Streamlit for easy interaction and rapid experimentation

**How to Use**
• Launch the app in your browser or local machine
•Enter input parameters: set values for fuel rate, airflows, and chemical components (CaO, SiO₂, Al₂O₃, Fe₂O₃)
•Click “Time Series Analysis” to run the model and generate predictions
•View real-time graphs showing how T_gas and T_solid change over time
•Analyze results, adjust inputs if needed, and run new simulations
•Export or save your data and configurations for later use
