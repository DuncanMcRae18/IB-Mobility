# Efficiency Colormap Generator for Intermediate Band Solar Cells
# Creates 2D/3D visualizations of efficiency landscapes

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pint import UnitRegistry


# Setup units
u = UnitRegistry()
h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi

T_sun = 6000 * u.kelvin
T_cell = 300 * u.kelvin

# Import core functions from original file
def power_input(T_sun, cutoff, fX):
    T_sun = T_sun.to('K').magnitude
    k = u.Quantity('boltzmann_constant').m_as('eV/K')
    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0, cutoff)
    P_in = fX * (2 * pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4')
    return P_in

def photon_flux(min, max, T, mu, fX):
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')
    flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)
    N = (fX * 2 * pi / ((h**3) * (c**2))) * flux * eV**3
    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):
    u_iv = u_cv - u_ci
    if E_i < E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
    else:
        flux_val = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1)
    flux = (flux_val + flux_conduct).magnitude
    return flux

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):
    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv * eV, 1)
    
    if E_i <= E_c:
        inter_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
        inter_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)
    else:
        inter_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1) 
        inter_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)
    
    flux = conduct_flux + inter_flux
    J = q * flux
    return J

def calculate_efficiency(E_g, E_i, u_cv, u_ci, int_max, fX):
    """Calculate efficiency for given parameters"""
    try:
        E_c = E_g - E_i
        u_iv = u_cv - u_ci
        
        # Basic validation
        if E_i <= 0 or E_c <= 0 or u_cv <= 0 or u_ci <= 0 or u_iv <= 0:
            return 0.0
        if u_ci >= u_cv or u_ci >= E_c.m_as('eV') or u_iv >= E_i.m_as('eV'):
            return 0.0
        
        # Check intermediate band current balance
        ib_current = IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)
        if abs(ib_current) > 1e-1:
            return 0.0
        
        # Calculate efficiency
        P_in = power_input(T_sun, int_max, fX)
        J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
        P_out = J * u_cv * eV / q
        efficiency = P_out / P_in
        
        if 0 < efficiency.magnitude < 1:
            return efficiency.magnitude * 100  # Return as percentage
        else:
            return 0.0
            
    except Exception as e:
        return 0.0

def generate_random_efficiency_data(n_points=15000, E_g_range=(0.7, 4.5), 
                                   E_i_fraction_range=(0.05, 0.95), concentration_factor=1, 
                                   smart_sampling=True):
    """Generate random efficiency data points"""
    
    print(f"Generating {n_points} random efficiency data points...")
    
    # Concentration factor setup
    f = 1/(2*pi*14000)
    fX = f * concentration_factor
    int_max = 10
    
    # Random parameter generation
    np.random.seed(42)  # For reproducibility
    
    results = []
    valid_points = 0
    attempts = 0
    max_attempts = n_points * 15  # More attempts for higher quality
    
    # Smart sampling: focus more points around known peak regions
    if smart_sampling:
        peak_regions = [
            {'ei_frac': (0.25, 0.35), 'weight': 0.4},  # Peak 1
            {'ei_frac': (0.65, 0.75), 'weight': 0.4},  # Peak 2
            {'ei_frac': (0.35, 0.65), 'weight': 0.15}, # Between peaks
            {'ei_frac': (0.05, 0.25), 'weight': 0.025}, # Low E_i
            {'ei_frac': (0.75, 0.95), 'weight': 0.025}  # High E_i
        ]
    
    while valid_points < n_points and attempts < max_attempts:
        attempts += 1
        
        # Generate E_g with slight bias toward optimal range (1.0-3.0 eV)
        if np.random.random() < 0.7:  # 70% in optimal range
            E_g_val = np.random.uniform(1.0, 3.0)
        else:
            E_g_val = np.random.uniform(E_g_range[0], E_g_range[1])
        E_g = E_g_val * eV
        
        # Smart E_i sampling based on regions
        if smart_sampling:
            region_choice = np.random.choice(len(peak_regions), 
                                           p=[r['weight'] for r in peak_regions])
            selected_region = peak_regions[region_choice]
            E_i_fraction = np.random.uniform(selected_region['ei_frac'][0], 
                                           selected_region['ei_frac'][1])
        else:
            E_i_fraction = np.random.uniform(E_i_fraction_range[0], E_i_fraction_range[1])
            
        E_i = E_i_fraction * E_g
        
        # More sophisticated u_cv and u_ci generation
        # u_cv should be reasonable fraction of E_g
        u_cv = np.random.beta(2, 2) * (0.8 * E_g_val) + 0.15 * E_g_val  # Beta distribution for better spread
        
        # u_ci should be smaller than u_cv and reasonable for the band structure
        max_uci = min(0.8 * u_cv, 0.6 * E_g_val)
        u_ci = np.random.uniform(0.05 * E_g_val, max_uci)
        
        # Calculate efficiency
        eff = calculate_efficiency(E_g, E_i, u_cv, u_ci, int_max, fX)
        
        if eff > 0:  # Only keep valid efficiencies
            results.append({
                'E_g': E_g_val,
                'E_i': E_i.m_as('eV'),
                'E_i_fraction': E_i_fraction,
                'u_cv': u_cv,
                'u_ci': u_ci,
                'efficiency': eff
            })
            valid_points += 1
            
        if attempts % 1000 == 0:
            print(f"  Attempts: {attempts}, Valid points: {valid_points}")
    
    print(f"Generated {valid_points} valid efficiency points from {attempts} attempts")
    return pd.DataFrame(results)

def advanced_interpolation(df, method='rbf', resolution=100):
    """Advanced interpolation with multiple methods and extrapolation"""
    
    # Prepare data
    points = np.column_stack((df['E_g'].values, df['E_i'].values))
    values = df['efficiency'].values
    
    # Create high-resolution grid
    E_g_min, E_g_max = df['E_g'].min(), df['E_g'].max()
    E_i_min, E_i_max = df['E_i'].min(), df['E_i'].max()
    
    # Extend grid slightly for extrapolation
    E_g_range = E_g_max - E_g_min
    E_i_range = E_i_max - E_i_min
    
    E_g_grid = np.linspace(E_g_min - 0.1*E_g_range, E_g_max + 0.1*E_g_range, resolution)
    E_i_grid = np.linspace(E_i_min - 0.1*E_i_range, E_i_max + 0.1*E_i_range, resolution)
    E_g_mesh, E_i_mesh = np.meshgrid(E_g_grid, E_i_grid)
    
    if method == 'rbf':
        # Radial Basis Function interpolation (best for smooth extrapolation)
        try:
            interpolator = RBFInterpolator(points, values, kernel='gaussian', epsilon=0.1)
            efficiency_interp = interpolator(np.column_stack([E_g_mesh.ravel(), E_i_mesh.ravel()]))
            efficiency_interp = efficiency_interp.reshape(E_g_mesh.shape)
        except:
            print("RBF failed, falling back to linear interpolation")
            efficiency_interp = griddata(points, values, (E_g_mesh, E_i_mesh), method='linear')
            
    elif method == 'clough_tocher':
        # Clough-Tocher interpolation (good for complex surfaces)
        interpolator = CloughTocher2DInterpolator(points, values)
        efficiency_interp = interpolator(E_g_mesh, E_i_mesh)
        
    elif method == 'linear_extrapolate':
        # Linear interpolation with nearest-neighbor extrapolation
        efficiency_interp = griddata(points, values, (E_g_mesh, E_i_mesh), 
                                   method='linear', fill_value=np.nan)
        # Fill NaN values with nearest neighbor for extrapolation
        mask = np.isnan(efficiency_interp)
        if np.any(mask):
            efficiency_nearest = griddata(points, values, (E_g_mesh, E_i_mesh), method='nearest')
            efficiency_interp[mask] = efficiency_nearest[mask]
            
    elif method == 'cubic_smooth':
        # Cubic interpolation with smoothing
        efficiency_interp = griddata(points, values, (E_g_mesh, E_i_mesh), method='cubic')
        # Smooth the result to reduce artifacts
        efficiency_interp = gaussian_filter(efficiency_interp, sigma=0.8)
        
    else:  # Default linear
        efficiency_interp = griddata(points, values, (E_g_mesh, E_i_mesh), method='linear')
    
    # Ensure physical bounds (efficiency between 0 and theoretical max ~68%)
    efficiency_interp = np.clip(efficiency_interp, 0, 68)
    
    return E_g_mesh, E_i_mesh, efficiency_interp

def create_efficiency_colormap(df, plot_type='E_g_vs_E_i', interpolation_method='rbf'):
    """Create colormap visualization of efficiency data"""
    
    if plot_type == 'E_g_vs_E_i':
        # E_g vs E_i colormap
        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter = ax.scatter(df['E_g'], df['E_i'], c=df['efficiency'], 
                           cmap='viridis', s=20, alpha=0.7, edgecolors='none')
        
        # Add theoretical peak lines
        E_g_line = np.linspace(df['E_g'].min(), df['E_g'].max(), 100)
        ax.plot(E_g_line, 0.3 * E_g_line, 'r--', alpha=0.7, linewidth=2, label='E_i = 0.3×E_g (Peak 1)')
        ax.plot(E_g_line, 0.7 * E_g_line, 'b--', alpha=0.7, linewidth=2, label='E_i = 0.7×E_g (Peak 2)')
        
        ax.set_xlabel('Bandgap Energy E_g (eV)', fontsize=12)
        ax.set_ylabel('Intermediate Band Energy E_i (eV)', fontsize=12)
        ax.set_title('Intermediate Band Solar Cell Efficiency Landscape', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Efficiency (%)')
        cbar.ax.tick_params(labelsize=11)
        
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
    elif plot_type == 'E_i_fraction':
        # E_g vs E_i/E_g fraction
        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter = ax.scatter(df['E_g'], df['E_i_fraction'], c=df['efficiency'], 
                           cmap='plasma', s=20, alpha=0.7, edgecolors='none')
        
        # Mark theoretical peaks
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Peak 1 (E_i/E_g = 0.3)')
        ax.axhline(y=0.7, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Peak 2 (E_i/E_g = 0.7)')
        
        ax.set_xlabel('Bandgap Energy E_g (eV)', fontsize=12)
        ax.set_ylabel('E_i/E_g Ratio', fontsize=12)
        ax.set_title('Efficiency vs Intermediate Band Fraction', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Efficiency (%)')
        cbar.ax.tick_params(labelsize=11)
        
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
    elif plot_type == 'contour':
        # Advanced contour plot using superior interpolation
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Use advanced interpolation
        E_g_mesh, E_i_mesh, efficiency_interp = advanced_interpolation(df, method=interpolation_method, resolution=80)
        
        # Create contour plot
        contour_filled = ax.contourf(E_g_mesh, E_i_mesh, efficiency_interp, 
                                   levels=20, cmap='RdYlBu_r')
        contour_lines = ax.contour(E_g_mesh, E_i_mesh, efficiency_interp, 
                                 levels=20, colors='black', alpha=0.4, linewidths=0.5)
        
        # Add theoretical peak lines
        E_g_line = np.linspace(df['E_g'].min(), df['E_g'].max(), 100)
        ax.plot(E_g_line, 0.3 * E_g_line, 'r--', alpha=0.8, linewidth=3, label='E_i = 0.3×E_g')
        ax.plot(E_g_line, 0.7 * E_g_line, 'b--', alpha=0.8, linewidth=3, label='E_i = 0.7×E_g')
        
        ax.set_xlabel('Bandgap Energy E_g (eV)', fontsize=12)
        ax.set_ylabel('Intermediate Band Energy E_i (eV)', fontsize=12)
        ax.set_title('Efficiency Contour Map', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(contour_filled, ax=ax, label='Efficiency (%)')
        cbar.ax.tick_params(labelsize=11)
        
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_3d_surface_plot(df, resolution=80, interpolation_method='rbf'):
    """Create 3D surface plot of efficiency with advanced interpolation"""
    
    print(f"Creating 3D surface plot with {interpolation_method} interpolation...")
    
    # Use advanced interpolation
    E_g_mesh, E_i_mesh, efficiency_interp = advanced_interpolation(df, method=interpolation_method, resolution=resolution)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_surface(E_g_mesh, E_i_mesh, efficiency_interp, 
                            cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Add theoretical peak lines as 3D lines
    E_g_line = np.linspace(df['E_g'].min(), df['E_g'].max(), 50)
    peak1_line = 0.3 * E_g_line
    peak2_line = 0.7 * E_g_line
    
    # Interpolate efficiency along peak lines for z-values
    points = np.column_stack((df['E_g'].values, df['E_i'].values))
    peak1_eff = griddata(points, df['efficiency'].values, 
                        np.column_stack((E_g_line, peak1_line)), method='linear')
    peak2_eff = griddata(points, df['efficiency'].values, 
                        np.column_stack((E_g_line, peak2_line)), method='linear')
    
    ax.plot(E_g_line, peak1_line, peak1_eff, 'r-', linewidth=4, alpha=0.9, label='Peak 1 Path')
    ax.plot(E_g_line, peak2_line, peak2_eff, 'b-', linewidth=4, alpha=0.9, label='Peak 2 Path')
    
    ax.set_xlabel('Bandgap Energy E_g (eV)', fontsize=12)
    ax.set_ylabel('Intermediate Band Energy E_i (eV)', fontsize=12)
    ax.set_zlabel('Efficiency (%)', fontsize=12)
    ax.set_title('3D Efficiency Surface', fontsize=14, fontweight='bold')
    
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=20, label='Efficiency (%)')
    ax.legend(loc='upper left', fontsize=11)
    
    plt.show()

def analyze_bimodal_peaks(df):
    """Analyze the bimodal nature of the efficiency landscape"""
    
    print("\n=== Bimodal Peak Analysis ===")
    
    # Define peak regions
    peak1_region = (df['E_i_fraction'] >= 0.25) & (df['E_i_fraction'] <= 0.35)
    peak2_region = (df['E_i_fraction'] >= 0.65) & (df['E_i_fraction'] <= 0.75)
    
    peak1_data = df[peak1_region]
    peak2_data = df[peak2_region]
    
    print(f"Peak 1 (E_i/E_g ≈ 0.3): {len(peak1_data)} points")
    print(f"  Max efficiency: {peak1_data['efficiency'].max():.2f}%")
    print(f"  Mean efficiency: {peak1_data['efficiency'].mean():.2f}%")
    print(f"  Best E_g: {peak1_data.loc[peak1_data['efficiency'].idxmax(), 'E_g']:.2f} eV")
    
    print(f"\nPeak 2 (E_i/E_g ≈ 0.7): {len(peak2_data)} points")
    print(f"  Max efficiency: {peak2_data['efficiency'].max():.2f}%")
    print(f"  Mean efficiency: {peak2_data['efficiency'].mean():.2f}%")
    print(f"  Best E_g: {peak2_data.loc[peak2_data['efficiency'].idxmax(), 'E_g']:.2f} eV")
    
    # Overall best
    overall_best = df.loc[df['efficiency'].idxmax()]
    print(f"\nOverall Best:")
    print(f"  Efficiency: {overall_best['efficiency']:.2f}%")
    print(f"  E_g: {overall_best['E_g']:.2f} eV")
    print(f"  E_i: {overall_best['E_i']:.2f} eV")
    print(f"  E_i/E_g: {overall_best['E_i_fraction']:.3f}")
    
    return peak1_data, peak2_data, overall_best

def create_interpolation_comparison(df):
    """Compare different interpolation methods side by side"""
    
    methods = ['linear_extrapolate', 'rbf', 'clough_tocher', 'cubic_smooth']
    method_names = ['Linear + Extrapolation', 'RBF Gaussian', 'Clough-Tocher', 'Cubic Smoothed']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        try:
            E_g_mesh, E_i_mesh, efficiency_interp = advanced_interpolation(df, method=method, resolution=100)
            
            # Create contour plot
            contour = axes[i].contourf(E_g_mesh, E_i_mesh, efficiency_interp, 
                                     levels=25, cmap='viridis')
            
            # Add original data points
            scatter = axes[i].scatter(df['E_g'], df['E_i'], c=df['efficiency'], 
                                    cmap='viridis', s=8, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add theoretical peak lines
            E_g_line = np.linspace(df['E_g'].min(), df['E_g'].max(), 100)
            axes[i].plot(E_g_line, 0.3 * E_g_line, 'r--', alpha=0.8, linewidth=2)
            axes[i].plot(E_g_line, 0.7 * E_g_line, 'b--', alpha=0.8, linewidth=2)
            
            axes[i].set_title(f'{name} Interpolation', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('E_g (eV)')
            axes[i].set_ylabel('E_i (eV)')
            axes[i].grid(True, alpha=0.3)
            
            plt.colorbar(contour, ax=axes[i], label='Efficiency (%)')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Method failed:\n{str(e)}', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{name} - Failed', fontsize=12)
    
    plt.suptitle('Interpolation Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_high_resolution_plot(df, resolution=150):
    """Create ultra-high resolution plot with best interpolation"""
    
    print(f"Creating ultra-high resolution plot ({resolution}x{resolution})...")
    
    # Use RBF for best smooth extrapolation
    E_g_mesh, E_i_mesh, efficiency_interp = advanced_interpolation(df, method='rbf', resolution=resolution)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # High-res contour plot
    contour1 = ax1.contourf(E_g_mesh, E_i_mesh, efficiency_interp, 
                           levels=50, cmap='plasma')
    contour_lines = ax1.contour(E_g_mesh, E_i_mesh, efficiency_interp, 
                              levels=20, colors='white', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f%%')
    
    # Original data points
    scatter1 = ax1.scatter(df['E_g'], df['E_i'], c='black', s=4, alpha=0.6)
    
    # Theoretical lines
    E_g_line = np.linspace(E_g_mesh.min(), E_g_mesh.max(), 100)
    ax1.plot(E_g_line, 0.3 * E_g_line, 'cyan', linewidth=3, alpha=0.9, label='E_i = 0.3×E_g')
    ax1.plot(E_g_line, 0.7 * E_g_line, 'yellow', linewidth=3, alpha=0.9, label='E_i = 0.7×E_g')
    
    ax1.set_title('Ultra-High Resolution Efficiency Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bandgap Energy E_g (eV)', fontsize=12)
    ax1.set_ylabel('Intermediate Band Energy E_i (eV)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)
    
    plt.colorbar(contour1, ax=ax1, label='Efficiency (%)')
    
    # Cross-sections at peak efficiency E_g values
    best_point = df.loc[df['efficiency'].idxmax()]
    best_Eg = best_point['E_g']
    
    # Find nearest E_g index in mesh
    Eg_idx = np.argmin(np.abs(E_g_mesh[0, :] - best_Eg))
    
    # Extract cross-section
    cross_section_eff = efficiency_interp[:, Eg_idx]
    cross_section_Ei = E_i_mesh[:, Eg_idx]
    
    ax2.plot(cross_section_Ei, cross_section_eff, 'b-', linewidth=3, label=f'E_g = {best_Eg:.2f} eV')
    ax2.scatter(df[df['E_g'].between(best_Eg-0.1, best_Eg+0.1)]['E_i'], 
               df[df['E_g'].between(best_Eg-0.1, best_Eg+0.1)]['efficiency'], 
               c='red', s=30, alpha=0.8, label='Data points')
    
    # Mark theoretical peaks
    ax2.axvline(0.3 * best_Eg, color='cyan', linestyle='--', alpha=0.7, label='Peak 1 position')
    ax2.axvline(0.7 * best_Eg, color='yellow', linestyle='--', alpha=0.7, label='Peak 2 position')
    
    ax2.set_title(f'Efficiency Cross-section at E_g = {best_Eg:.2f} eV', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intermediate Band Energy E_i (eV)', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Advanced Intermediate Band Solar Cell Efficiency Mapping ===\n")
    
    # Generate high-density efficiency data with smart sampling
    print("Step 1: Generating high-density efficiency data...")
    df_X1 = generate_random_efficiency_data(n_points=12000, concentration_factor=1, smart_sampling=True)
    
    print(f"\nGenerated {len(df_X1)} valid data points")
    print(f"Efficiency range: {df_X1['efficiency'].min():.2f}% - {df_X1['efficiency'].max():.2f}%")
    print(f"E_g range: {df_X1['E_g'].min():.2f} - {df_X1['E_g'].max():.2f} eV")
    
    print("\nStep 2: Creating advanced visualizations...")
    
    # Basic scatter plots
    print("  Creating enhanced scatter plots...")
    create_efficiency_colormap(df_X1, plot_type='E_g_vs_E_i')
    create_efficiency_colormap(df_X1, plot_type='E_i_fraction')
    
    # Advanced interpolated plots
    print("  Creating advanced contour plot with RBF interpolation...")
    create_efficiency_colormap(df_X1, plot_type='contour', interpolation_method='rbf')
    
    print("  Creating 3D surface with advanced interpolation...")
    create_3d_surface_plot(df_X1, resolution=100, interpolation_method='rbf')
    
    # Comparison of interpolation methods
    print("  Creating interpolation method comparison...")
    create_interpolation_comparison(df_X1)
    
    # Ultra-high resolution plot
    print("  Creating ultra-high resolution analysis...")
    create_high_resolution_plot(df_X1, resolution=120)
    
    # Analyze bimodal peaks
    print("\nStep 3: Analyzing bimodal peaks...")
    peak1_data, peak2_data, best_point = analyze_bimodal_peaks(df_X1)
    
    print("\n=== Advanced Analysis Complete ===")
    print("Generated comprehensive high-resolution efficiency landscape!")
    
    # Save data
    save_data = input("\nSave efficiency data to CSV? (y/n): ").lower().strip()
    if save_data == 'y':
        df_X1.to_csv("efficiency_X1_advanced.csv", index=False)
        print("Data saved to efficiency_X1_advanced.csv")