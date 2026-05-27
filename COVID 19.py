import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_covid_sir(population, initial_infected, contact_rate, recovery_rate, days):
    """
    Simulates the spread of COVID-19 using the scientific SIR differential model equations.
    - population (N): Total number of people
    - contact_rate (beta): Average number of adequate contacts per infected individual per day
    - recovery_rate (gamma): 1 / days_to_recover
    """
    # Initialize arrays to store data over time
    S = np.zeros(days)  # Susceptible
    I = np.zeros(days)  # Infected (Active cases)
    R = np.zeros(days)  # Recovered/Removed
    
    # Initial conditions
    I[0] = initial_infected
    R[0] = 0
    S[0] = population - initial_infected - R[0]
    
    # Run simulation using Euler's method for scientific differential approximations
    for t in range(1, days):
        # Calculate current daily rates of change
        newly_infected = (contact_rate * S[t-1] * I[t-1]) / population
        newly_recovered = recovery_rate * I[t-1]
        
        # Update current state vectors
        S[t] = S[t-1] - newly_infected
        I[t] = I[t-1] + newly_infected - newly_recovered
        R[t] = R[t-1] + newly_recovered
        
    # Wrap results inside a Pandas DataFrame for structured data manipulation
    data_dict = {
        'Day': np.arange(days),
        'Susceptible': S,
        'Infected_Active': I,
        'Recovered_Immune': R
    }
    df = pd.DataFrame(data_dict)
    
    # Feature Engineering: Compute derived tracking metrics
    df['Total_Confirmed_Cases'] = df['Infected_Active'] + df['Recovered_Immune']
    df['Daily_New_Cases'] = df['Total_Confirmed_Cases'].diff().fillna(0)
    
    return df

def analyze_and_plot(df, r_zero):
    """
    Performs data science calculations on the simulation output and plots charts.
    """
    # 1. Scientific Data Analytics Calculations
    peak_active_idx = df['Infected_Active'].idxmax()
    peak_day = df.loc[peak_active_idx, 'Day']
    peak_count = df.loc[peak_active_idx, 'Infected_Active']
    total_impacted = df['Total_Confirmed_Cases'].iloc[-1]
    
    print("====================================================")
    print("      COVID-19 SCIENTIFIC SIMULATION REPORT         ")
    print("====================================================")
    print(f"Basic Reproduction Number (R0) : {r_zero:.2f}")
    print(f"Peak Outbreak Day              : Day {int(peak_day)}")
    print(f"Max Active Infections at Peak  : {int(peak_count):,}")
    print(f"Total Combined Infections      : {int(total_impacted):,}")
    print("====================================================\n")
    
    # Display snippet of the compiled matrix
    print("First 5 days of data framework output:")
    print(df.head(), "\n")

    # 2. Advanced Data Visualization (Matplotlib backend)
    plt.figure(figsize=(12, 6))
    
    # Plotting Compartment Curves
    plt.plot(df['Day'], df['Susceptible'], label='Susceptible (Uninfected)', color='blue', lw=2)
    plt.plot(df['Day'], df['Infected_Active'], label='Infected (Active Cases)', color='red', lw=3)
    plt.plot(df['Day'], df['Recovered_Immune'], label='Recovered / Immune', color='green', lw=2)
    
    # Draw reference marker line highlighting the hospital capacity strain threshold
    plt.axvline(x=peak_day, color='gray', linestyle='--', alpha=0.7, label=f'Outbreak Peak (Day {int(peak_day)})')
    
    # Chart styling configurations
    plt.title('COVID-19 Scientific Modeling: Compartmental Dynamics (SIR)', fontsize=14, fontweight='bold')
    plt.xlabel('Days Since Patient Zero Outbreak', fontsize=11)
    plt.ylabel('Population Matrix Count', fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    
    # Show plot directly to window framework
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION FRAMEWORK ---
if __name__ == "__main__":
    # Parameters optimized for a simulated regional community setup
    TOTAL_POPULATION = 100_000         # Total regional community cohort size (N)
    PATIENT_ZERO_COUNT = 10            # Initial active infections starting the vector
    TIMELINE_DAYS = 120                # Time duration for computational processing steps
    
    # Pathogen structural variables
    BETA = 0.35                        # High contact rate (representing low social distancing)
    GAMMA = 1/14                       # Recovery rate (assuming standard 14-day recovery window)
    
    # Basic Reproduction Number calculation (R0 = Beta / Gamma)
    # R0 > 1 means an epidemic expands; R0 < 1 means the pathogen dies out
    R0 = BETA / GAMMA                  
    
    # Process computational pipeline modeling
    covid_dataframe = simulate_covid_sir(TOTAL_POPULATION, PATIENT_ZERO_COUNT, BETA, GAMMA, TIMELINE_DAYS)
    
    # Execute analysis engine and plot visual rendering frames
    analyze_and_plot(covid_dataframe, R0)