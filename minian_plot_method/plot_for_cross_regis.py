import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import xarray as xr
from scipy.signal import find_peaks
import os

# --- Global Data Manager Instance ---
# This variable will hold the single instance of MinianDataManager to ensure data is loaded only once
_global_minian_data_manager = None

class MinianDataManager:
    """
    Manages the loading of Minian data files.
    This is a standard class, and its instance will be held by a global variable.
    All file path parameters are now mandatory, with no default values.
    Loading of YrA-C Std CSV files has been added.
    """
    def __init__(self, mappings_csv_path: str,
                 session1_nc_path: str,
                 session2_nc_path: str,
                 cents_csv_path: str = None,
                 std_csv_path1: str = None, # New parameter, defaults to None to allow optional provision
                 std_csv_path2: str = None): # New parameter, defaults to None to allow optional provision
        
        self.mappings_csv_path = mappings_csv_path
        self.cents_csv_path = cents_csv_path
        self.session1_nc_path = session1_nc_path
        self.session2_nc_path = session2_nc_path
        self.std_csv_path1 = std_csv_path1 # Store the new std path
        self.std_csv_path2 = std_csv_path2 # Store the new std path

        self.mappings_df = None
        self.ds1 = None
        self.ds2 = None
        self.std_df1 = None # Store std data
        self.std_df2 = None # Store std data
        self.session1_name = 'session1' # Default, will be overwritten if available in mappings
        self.session2_name = 'session2' # Default, will be overwritten if available in mappings
        self.image_height = 480 # Default, will be overwritten if available in .nc
        self.image_width = 752 # Default, will be overwritten if available in .nc
        
        self.load_data() # Load data upon initialization

    def load_data(self):
        """Loads all necessary data files."""
        print(f"Loading Minian data from the following paths:")
        print(f"   Mappings: {self.mappings_csv_path}")
        print(f"   Cents: {self.cents_csv_path}")
        print(f"   Session 1 .nc: {self.session1_nc_path}")
        print(f"   Session 2 .nc: {self.session2_nc_path}")
        if self.std_csv_path1:
            print(f"   Session 1 Std: {self.std_csv_path1}")
        if self.std_csv_path2:
            print(f"   Session 2 Std: {self.std_csv_path2}")
        print("-" * 30)

        # Load CSV files
        try:
            self.mappings_df = pd.read_csv(self.mappings_csv_path)
            print("Mappings CSV loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: Mappings CSV file not found. Please check path: {e}")
            self.mappings_df = None
        except Exception as e:
            print(f"An error occurred during Mappings CSV loading: {e}")
            self.mappings_df = None

        # Load Cents CSV files
        if self.cents_csv_path: # Only load if path is provided
            try:
                self.cents_df = pd.read_csv(self.cents_csv_path)
                print("Cents CSV loaded successfully.")
            except FileNotFoundError as e:
                print(f"Error: Cents CSV file not found. Please check path: {e}")
                self.cents_df = None
            except Exception as e:
                print(f"An error occurred during Cents CSV loading: {e}")
                self.cents_df = None
        else:
            self.cents_df = None
            print("No cents.csv path provided. Centroid plotting will be unavailable.")

        # Load .nc files
        try:
            self.ds1 = xr.open_dataset(self.session1_nc_path)
            self.ds2 = xr.open_dataset(self.session2_nc_path)
            print(".nc files loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: Minian .nc file not found. Please check path: {e}")
            self.ds1, self.ds2 = None, None
        except Exception as e:
            print(f"An error occurred during .nc loading: {e}")
            self.ds1, self.ds2 = None, None
        
        # --- MODIFIED PREPROCESSING STEP: Truncate C traces to match shortest session length ---
        if self.ds1 is not None and self.ds2 is not None:
            frames1 = self.ds1.sizes['frame']
            frames2 = self.ds2.sizes['frame']

            if frames1 != frames2:
                min_frames = min(frames1, frames2)
                print(f"Detected different frame lengths: Session 1 ({frames1} frames), Session 2 ({frames2} frames).")
                print(f"Truncating both datasets to common length: {min_frames} frames.")

                # Apply isel to the entire Dataset to ensure all variables with 'frame' dimension are truncated
                if frames1 > min_frames:
                    self.ds1 = self.ds1.isel(frame=slice(0, min_frames))
                    print(f"Session 1 dataset truncated from {frames1} to {min_frames} frames.")
                
                if frames2 > min_frames:
                    self.ds2 = self.ds2.isel(frame=slice(0, min_frames))
                    print(f"Session 2 dataset truncated from {frames2} to {min_frames} frames.")
                
                print(f"All relevant data standardized to {min_frames} frames.")
            else:
                print(f"Frame lengths are already identical ({frames1} frames). No truncation needed.")
        else:
            print("Skipping frame length synchronization as one or both .nc files failed to load.")
        # --- END MODIFIED PREPROCESSING STEP ---

        # Load Std CSV files
        if self.std_csv_path1:
            try:
                self.std_df1 = pd.read_csv(self.std_csv_path1)
                print(f"Session 1 Std CSV (from {self.std_csv_path1}) loaded successfully.")
            except FileNotFoundError as e:
                print(f"Warning: Session 1 Std CSV file not found. C traces will not be normalized. {e}")
                self.std_df1 = None
            except Exception as e:
                print(f"Warning: An error occurred while loading Session 1 Std CSV file. C traces will not be normalized. {e}")
                self.std_df1 = None
        else:
            self.std_df1 = None
            print("No std_csv_path1 provided. Session 1 C traces will not be normalized by STD.")
        
        if self.std_csv_path2:
            try:
                self.std_df2 = pd.read_csv(self.std_csv_path2)
                print(f"Session 2 Std CSV (from {self.std_csv_path2}) loaded successfully.")
            except FileNotFoundError as e:
                print(f"Warning: Session 2 Std CSV file not found. C traces will not be normalized. {e}")
                self.std_df2 = None
            except Exception as e:
                print(f"Warning: An error occurred while loading Session 2 Std CSV file. C traces will not be normalized. {e}")
                self.std_df2 = None
        else:
            self.std_df2 = None
            print("No std_csv_path2 provided. Session 2 C traces will not be normalized by STD.")


        # Extract session names (only if mappings_df loaded successfully and has enough columns)
        if self.mappings_df is not None and not self.mappings_df.empty:
            try:
                # Assuming the first row contains session names in the second and third columns
                if self.mappings_df.shape[1] > 2: # Ensure at least 3 columns to extract session names
                    self.session1_name = str(self.mappings_df.iloc[0, 1]).strip()
                    self.session2_name = str(self.mappings_df.iloc[0, 2]).strip()
                    print(f"Session 1 name: {self.session1_name}")
                    print(f"Session 2 name: {self.session2_name}")
                else:
                    print("Warning: mappings.csv has insufficient columns to extract session names. Using default names.")
            except Exception as e:
                print(f"Error extracting session names: {e}. Using default names.")
        else:
            print("Warning: mappings.csv not loaded or is empty, cannot extract session names. Using default names.")

        # Get image dimensions (only if ds1 loaded successfully)
        if self.ds1 and 'A' in self.ds1.variables and 'height' in self.ds1['A'].dims and 'width' in self.ds1['A'].dims:
            self.image_height = self.ds1['A'].sizes['height']
            self.image_width = self.ds1['A'].sizes['width']
        else:
            print("Warning: Could not get image dimensions from .nc file. Using default dimensions.")

    def get_data(self):
        """Returns all loaded data."""
        return {
            "mappings_df": self.mappings_df,
            "cents_df": self.cents_df,
            "ds1": self.ds1,
            "ds2": self.ds2,
            "std_df1": self.std_df1, # Return std data
            "std_df2": self.std_df2, # Return std data
            "session1_name": self.session1_name,
            "session2_name": self.session2_name,
            "image_height": self.image_height,
            "image_width": self.image_width
        }
# --- Helper Functions (Access global instance to get data) ---

def _detect_peaks(trace, prominence, distance):
    """
    Detects peaks in the given trace.
    Args:
        trace (np.array): Trace data to detect peaks from.
        prominence (float): The relative prominence of peaks.
        distance (int): Minimum horizontal distance between peaks.
    Returns:
        tuple: A tuple containing the array of peak indices and a dictionary of peak properties.
    """
    peaks, properties = find_peaks(trace, prominence=prominence, distance=distance, height=None) 
    return peaks, properties

def _get_spike_stats(trace, prominence, distance, frames):
    """
    Calculates spike (or "peak") statistics for a given trace.
    Args:
        trace (np.array): Trace data to analyze.
        prominence (float): The relative prominence of peaks.
        distance (int): Minimum horizontal distance between peaks.
        frames (int): Total number of frames in the trace.
    Returns:
        tuple: A tuple containing (spike_count, list of spike indices, list of spike values, max_value, mean_value, std_value, spike_rate).
    """
    peaks, properties = _detect_peaks(trace, prominence, distance)
    # Safely get 'peak_heights' using .get(), returning an empty list if not present
    values = trace[peaks]
    spike_count = len(values)
    spike_rate = spike_count / frames * 100 if frames > 0 else 0
    max_val = np.max(values) if spike_count > 0 else 0
    mean_val = np.mean(values) if spike_count > 0 else 0
    std_val = np.std(values) if spike_count > 0 else 0
    
    return spike_count, peaks.tolist(), values, max_val, mean_val, std_val, spike_rate

# ... (MinianDataManager, _detect_peaks, _get_spike_stats 定义不变)

def _plot_minian_C_traces_and_mappings(
    cell1_id: int,
    cell2_id: int,
    show_overlayed_traces: bool = True,
    show_individual_traces: bool = True,
    detect_spikes: bool = True,
    spike_prominence: float = 4.0,
    spike_distance: int = 5,
    enable_normalization: bool = True,
    show_peak_distribution: bool = True,
    histogram_bins: int = 50,
    show_full_fov: bool = True,
    show_local_view: bool = True,
    show_cell1_individual_map: bool = True,
    show_cell2_individual_map: bool = True,
    show_overlap_map_local: bool = True,
    local_padding: int = 80,
    min_crop_size: int = 150,
    save_path_prefix: str = None,
    mapping_idx: int = None
):
    plt.rcParams['axes.unicode_minus'] = False

    global _global_minian_data_manager
    if _global_minian_data_manager is None:
        print("Error: MinianDataManager has not been initialized. Please call interactive_minian_analyzer first with file paths.")
        return
    data = _global_minian_data_manager.get_data()

    ds1 = data["ds1"]
    ds2 = data["ds2"]
    std_df1 = data["std_df1"]
    std_df2 = data["std_df2"]
    cents_df = data["cents_df"]
    session1_name = data["session1_name"]
    session2_name = data["session2_name"]
    image_height = data["image_height"]
    image_width = data["image_width"]

    _save_only_mode = (save_path_prefix is not None)

    if ds1 is None or ds2 is None:
        if not _save_only_mode:
            print("Minian .nc data not loaded, cannot plot C traces or spatial maps.")
        return

    # --- C Trace Plotting Logic ---
    if show_overlayed_traces or show_individual_traces:
        try:
            c_trace1 = ds1['C'].sel(unit_id=cell1_id).values
            c_trace2 = ds2['C'].sel(unit_id=cell2_id).values
        except (KeyError, IndexError) as e:
            if not _save_only_mode:
                print(f"Error: C trace not found for unit ID {e}. Please check if the ID exists in the .nc files.")
            return

        frames1 = len(c_trace1)
        frames2 = len(c_trace2)

        c_trace1_plot = c_trace1
        c_trace2_plot = c_trace2

        if enable_normalization:
            std_val1 = 1.0
            if std_df1 is not None and 'unit_id' in std_df1.columns and 'YRA_C_Std' in std_df1.columns:
                std_row1 = std_df1[std_df1['unit_id'] == cell1_id]['YRA_C_Std']
                if not std_row1.empty and std_row1.iloc[0] > 0:
                    std_val1 = std_row1.iloc[0]
            c_trace1_plot = c_trace1 / std_val1

            std_val2 = 1.0
            if std_df2 is not None and 'unit_id' in std_df2.columns and 'YRA_C_Std' in std_df2.columns:
                std_row2 = std_df2[std_df2['unit_id'] == cell2_id]['YRA_C_Std']
                if not std_row2.empty and std_row2.iloc[0] > 0:
                    std_val2 = std_row2.iloc[0]
            c_trace2_plot = c_trace2 / std_val2

        s1_spike_count, s1_peak_indices, s1_peak_values, _, _, _, _ = \
            _get_spike_stats(c_trace1_plot, spike_prominence, spike_distance, frames1)
        s2_spike_count, s2_peak_indices, s2_peak_values, _, _, _, _ = \
            _get_spike_stats(c_trace2_plot, spike_prominence, spike_distance, frames2)

        if show_overlayed_traces:
            fig_overlay = plt.figure(figsize=(12, 5))
            plt.plot(c_trace1_plot, label=f'{session1_name} C-Trace (ID: {cell1_id})', color='blue', alpha=0.7)
            plt.plot(c_trace2_plot, label=f'{session2_name} C-Trace (ID: {cell2_id})', color='red', alpha=0.7)
            plt.axhline(y=spike_prominence, color='green', linestyle='--', label=f'Prominence Threshold ({spike_prominence:.2f})')
            if detect_spikes:
                plt.plot(s1_peak_indices, c_trace1_plot[s1_peak_indices], "o", markersize=8, color='darkblue', alpha=0.7, label=f'{session1_name} Peaks')
                plt.plot(s2_peak_indices, c_trace2_plot[s2_peak_indices], "x", markersize=8, color='darkred', alpha=0.7, label=f'{session2_name} Peaks')
            plt.title(f'Overlayed C Matrix Traces (ID: {cell1_id} vs {cell2_id})')
            plt.xlabel('Frame')
            plt.ylabel('C Matrix Signal Intensity' + (' (STD Normalized)' if enable_normalization else ''))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if _save_only_mode:
                fig_overlay.savefig(os.path.join(save_path_prefix, f'M{mapping_idx}-S1-{cell1_id}-S2-{cell2_id}_Overlayed_CTraces.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_overlay)
            else:
                plt.show()

        if show_individual_traces:
            fig_individual, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            axs[0].plot(c_trace1_plot, label=f'{session1_name} C-Trace (ID: {cell1_id})', color='blue')
            if detect_spikes:
                axs[0].plot(s1_peak_indices, c_trace1_plot[s1_peak_indices], "o", markersize=8, color='darkblue', alpha=0.7, label='Detected Peaks')
            axs[0].axhline(y=spike_prominence, color='green', linestyle='--', label=f'Prominence Threshold ({spike_prominence:.2f})')
            axs[0].set_title(f'{session1_name} C Matrix Trace (ID: {cell1_id})')
            axs[0].set_ylabel('C Matrix Signal Intensity' + (' (STD Normalized)' if enable_normalization else ''))
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(c_trace2_plot, label=f'{session2_name} C-Trace (ID: {cell2_id})', color='red')
            if detect_spikes:
                axs[1].plot(s2_peak_indices, c_trace2_plot[s2_peak_indices], "x", markersize=8, color='darkred', alpha=0.7, label='Detected Peaks')
            axs[1].axhline(y=spike_prominence, color='green', linestyle='--', label=f'Prominence Threshold ({spike_prominence:.2f})')
            axs[1].set_title(f'{session2_name} C Matrix Trace (ID: {cell2_id})')
            axs[1].set_xlabel('Frame')
            axs[1].set_ylabel('C Matrix Signal Intensity' + (' (STD Normalized)' if enable_normalization else ''))
            axs[1].legend()
            axs[1].grid(True)
            plt.tight_layout()
            if _save_only_mode:
                fig_individual.savefig(os.path.join(save_path_prefix, f'M{mapping_idx}-S1-{cell1_id}-S2-{cell2_id}_Individual_CTraces.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_individual)
            else:
                plt.show()

    # --- Frequency Distribution Plotting Logic ---
    if show_peak_distribution:
        fig_dist, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        magnitude_label = 'Peak Magnitude' + (' (STD Normalized)' if enable_normalization else '')

        if len(s1_peak_values) > 0:
            axs[0].hist(s1_peak_values, bins=histogram_bins, color='blue', alpha=0.7, label=f'{session1_name} (N={len(s1_peak_values)} peaks)')
            axs[0].set_title(f'C Peak Magnitude Distribution for {session1_name} (ID: {cell1_id})')
            axs[0].set_xlabel(magnitude_label)
            axs[0].set_ylabel('Frequency')
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].set_title(f'No Peaks Detected for {session1_name} (ID: {cell1_id})')
            axs[0].set_xlabel(magnitude_label)
            axs[0].set_ylabel('Frequency')

        if len(s2_peak_values) > 0:
            axs[1].hist(s2_peak_values, bins=histogram_bins, color='red', alpha=0.7, label=f'{session2_name} (N={len(s2_peak_values)} peaks)')
            axs[1].set_title(f'C Peak Magnitude Distribution for {session2_name} (ID: {cell2_id})')
            axs[1].set_xlabel(magnitude_label)
            axs[1].set_ylabel('Frequency')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_title(f'No Peaks Detected for {session2_name} (ID: {cell2_id})')
            axs[1].set_xlabel(magnitude_label)
            axs[1].set_ylabel('Frequency')

        plt.tight_layout()
        if _save_only_mode:
            fig_dist.savefig(os.path.join(save_path_prefix, f'M{mapping_idx}-S1-{cell1_id}-S2-{cell2_id}_PeakMagnitudeDistribution.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_dist)
        else:
            plt.show()

    # --- Spatial Mapping Plotting Logic ---
# --- Spatial Mapping Plotting Logic ---
    if show_full_fov or show_local_view or show_cell1_individual_map or show_cell2_individual_map or show_overlap_map_local:
        cell1_spatial_map = None
        cell2_spatial_map = None

        try:
            with xr.open_dataset(_global_minian_data_manager.session1_nc_path) as ds1_map:
                if 'A' in ds1_map.variables and 'unit_id' in ds1_map['A'].dims:
                    cell1_spatial_map = ds1_map.A.sel(unit_id=cell1_id).values
                else:
                    raise ValueError(f"'A' matrix or 'unit_id' dimension not found in {_global_minian_data_manager.session1_nc_path}")

            with xr.open_dataset(_global_minian_data_manager.session2_nc_path) as ds2_map:
                if 'A' in ds2_map.variables and 'unit_id' in ds2_map['A'].dims:
                    cell2_spatial_map = ds2_map.A.sel(unit_id=cell2_id).values
                else:
                    raise ValueError(f"'A' matrix or 'unit_id' dimension not found in {_global_minian_data_manager.session2_nc_path}")

            cell1_spatial_map = cell1_spatial_map / np.max(cell1_spatial_map) if np.max(cell1_spatial_map) > 0 else cell1_spatial_map
            cell2_spatial_map = cell2_spatial_map / np.max(cell2_spatial_map) if np.max(cell2_spatial_map) > 0 else cell2_spatial_map

        except KeyError as ke:
            if not _save_only_mode:
                print(f"Warning: Spatial map for Cell ID {ke} not found in .nc file. Fallback to plotting centroids.")
            cell1_spatial_map = None
            cell2_spatial_map = None
        except FileNotFoundError as fnfe:
            if not _save_only_mode:
                print(f"Error: .nc file not found at specified path for spatial map: {fnfe}. Fallback to plotting centroids.")
            cell1_spatial_map = None
            cell2_spatial_map = None
        except Exception as e:
            if not _save_only_mode:
                print(f"Error extracting spatial map: {e}. Fallback to plotting centroids.")
            cell1_spatial_map = None
            cell2_spatial_map = None

        if cell1_spatial_map is not None and cell2_spatial_map is not None:
            num_spatial_plots = sum([show_full_fov, show_local_view, show_cell1_individual_map, show_cell2_individual_map, show_overlap_map_local])
            n_cols = 2 if num_spatial_plots >= 2 else 1
            n_rows = (num_spatial_plots + n_cols - 1) // n_cols
            fig_spatial, axes_spatial = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 9))
            if n_rows == 1 and n_cols == 1:
                axes_flat = [axes_spatial]
            else:
                axes_flat = axes_spatial.flatten()

            plot_idx_spatial = 0
            background_img_base = np.full((image_height, image_width), 0.8)

            local_view_valid = False
            cell1_pos_df = cents_df[(cents_df['unit_id'] == cell1_id) & (cents_df['session'] == session1_name)]
            cell2_pos_df = cents_df[(cents_df['unit_id'] == cell2_id) & (cents_df['session'] == session2_name)]

            if not cell1_pos_df.empty and not cell2_pos_df.empty:
                center_x = (cell1_pos_df['width'].iloc[0] + cell2_pos_df['width'].iloc[0]) / 2
                center_y = (cell1_pos_df['height'].iloc[0] + cell2_pos_df['height'].iloc[0]) / 2

                y_min_idx = max(0, int(center_y - local_padding))
                y_max_idx = min(image_height, int(center_y + local_padding))
                x_min_idx = max(0, int(center_x - local_padding))
                x_max_idx = min(image_width, int(center_x + local_padding))

                if (y_max_idx - y_min_idx) < min_crop_size:
                    y_center_temp = (y_min_idx + y_max_idx) / 2
                    y_min_idx = max(0, int(y_center_temp - min_crop_size / 2))
                    y_max_idx = min(image_height, int(y_center_temp + min_crop_size / 2))
                if (x_max_idx - x_min_idx) < min_crop_size:
                    x_center_temp = (x_min_idx + x_max_idx) / 2
                    x_min_idx = max(0, int(x_center_temp - min_crop_size / 2))
                    x_max_idx = min(image_width, int(x_center_temp + min_crop_size / 2))

                local_view_valid = True

            if show_full_fov and plot_idx_spatial < len(axes_flat):
                ax = axes_flat[plot_idx_spatial]
                ax.imshow(background_img_base, cmap='gray', alpha=0.5, origin='lower')
                ax.imshow(cell1_spatial_map, cmap='Blues', alpha=0.7, origin='lower', vmin=0, vmax=1)
                ax.imshow(cell2_spatial_map, cmap='Reds', alpha=0.7, origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Full FOV Heatmap for S1 ID {cell1_id} <-> S2 ID {cell2_id}')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.grid(True)
                ax.set_aspect('equal', adjustable='box')
                plot_idx_spatial += 1

            if show_cell1_individual_map and plot_idx_spatial < len(axes_flat):
                ax = axes_flat[plot_idx_spatial]
                ax.imshow(background_img_base, cmap='gray', alpha=0.5, origin='lower')
                ax.imshow(cell1_spatial_map, cmap='Blues', alpha=0.9, origin='lower', vmin=0, vmax=1)
                ax.set_title(f'{session1_name} Cell {cell1_id} Spatial Map')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.grid(True)
                ax.set_aspect('equal', adjustable='box')
                plot_idx_spatial += 1

            if show_cell2_individual_map and plot_idx_spatial < len(axes_flat):
                ax = axes_flat[plot_idx_spatial]
                ax.imshow(background_img_base, cmap='gray', alpha=0.5, origin='lower')
                ax.imshow(cell2_spatial_map, cmap='Reds', alpha=0.9, origin='lower', vmin=0, vmax=1)
                ax.set_title(f'{session2_name} Cell {cell2_id} Spatial Map')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.grid(True)
                ax.set_aspect('equal', adjustable='box')
                plot_idx_spatial += 1

            if show_local_view and local_view_valid and plot_idx_spatial < len(axes_flat):
                ax = axes_flat[plot_idx_spatial]
                cell1_spatial_map_local = cell1_spatial_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                cell2_spatial_map_local = cell2_spatial_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                background_local_slice = background_img_base[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                extent_local = [x_min_idx, x_max_idx, y_min_idx, y_max_idx]

                ax.imshow(background_local_slice, cmap='gray', alpha=0.5, origin='lower', extent=extent_local)
                ax.imshow(cell1_spatial_map_local, cmap='Blues', alpha=0.7, origin='lower', vmin=0, vmax=1, extent=extent_local)
                ax.imshow(cell2_spatial_map_local, cmap='Reds', alpha=0.7, origin='lower', vmin=0, vmax=1, extent=extent_local)
                ax.set_title('Local View Heatmap')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.grid(True)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(x_min_idx, x_max_idx)
                ax.set_ylim(y_min_idx, y_max_idx)
                plot_idx_spatial += 1
            elif show_local_view and not local_view_valid and not _save_only_mode:
                print("Warning: Centroid positions not found, local view cannot be accurately centered.")

            if show_overlap_map_local and local_view_valid and plot_idx_spatial < len(axes_flat):
                ax = axes_flat[plot_idx_spatial]
                cell1_local = cell1_spatial_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                cell2_local = cell2_spatial_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                extent_local = [x_min_idx, x_max_idx, y_min_idx, y_max_idx]

                activity_threshold = 0.05
                cell1_only_mask = (cell1_local > activity_threshold) & (cell2_local <= activity_threshold)
                cell2_only_mask = (cell2_local > activity_threshold) & (cell1_local <= activity_threshold)
                overlap_mask = (cell1_local > activity_threshold) & (cell2_local > activity_threshold)

                light_gray_value = 0.8
                composite_display = np.full((cell1_local.shape[0], cell1_local.shape[1], 3), light_gray_value)

                composite_display[cell1_only_mask, :] = [1.0, 0.0, 0.0]  # Red for cell1 only
                composite_display[cell2_only_mask, :] = [0.0, 0.0, 1.0]  # Blue for cell2 only
                composite_display[overlap_mask, :] = [0.0, 1.0, 0.0]    # Green for overlap

                composite_display = np.clip(composite_display, 0, 1)

                ax.imshow(composite_display, origin='lower', extent=extent_local)
                ax.set_title('Color Overlay Map (Local View)')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.grid(True)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(x_min_idx, x_max_idx)
                ax.set_ylim(y_min_idx, y_max_idx)

                from matplotlib.patches import Patch
                color_cell1_only_legend = [1.0, 0.0, 0.0]
                color_cell2_only_legend = [0.0, 0.0, 1.0]
                color_overlap_legend = [0.0, 1.0, 0.0]

                legend_elements = [
                    Patch(facecolor=color_cell1_only_legend, label=f'{session1_name} Only'),
                    Patch(facecolor=color_cell2_only_legend, label=f'{session2_name} Only'),
                    Patch(facecolor=color_overlap_legend, label='Overlap')
                ]
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
                plot_idx_spatial += 1
            elif show_overlap_map_local and not local_view_valid and not _save_only_mode:
                print("Warning: Centroid positions not found, local overlap view cannot be accurately centered.")

            
            elif show_overlap_map_local and not local_view_valid and not _save_only_mode:
                print("Warning: Centroid positions not found, monochrome distribution view cannot be accurately centered.")

            for i in range(plot_idx_spatial, len(axes_flat)):
                fig_spatial.delaxes(axes_flat[i])

            plt.tight_layout()
            if _save_only_mode:
                fig_spatial.savefig(os.path.join(save_path_prefix, f'M{mapping_idx}-S1-{cell1_id}-S2-{cell2_id}_SpatialMaps.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_spatial)
            else:
                plt.show()
    if not _save_only_mode:
        print("\n--- Selected Pair Plotting Complete ---")

def analyze_and_plot_c_data(
    params: dict
):
    print("\n--- Starting Comprehensive C Matrix Analysis and Plotting ---")

    global _global_minian_data_manager
    if _global_minian_data_manager is None:
        print("Error: MinianDataManager has not been initialized. Please call interactive_minian_analyzer first with file paths.")
        return
    data = _global_minian_data_manager.get_data()

    mappings_df = data["mappings_df"]
    ds1 = data["ds1"]
    ds2 = data["ds2"]
    std_df1 = data["std_df1"]
    std_df2 = data["std_df2"]
    session1_name = data["session1_name"]
    session2_name = data["session2_name"]

    if mappings_df is None or ds1 is None or ds2 is None:
        print("Required data not loaded, cannot proceed with analysis and plotting.")
        return

    peak_csv_path = params.get('peak_csv_path', 'c_trace_peak_stats.csv')
    mapping_class_csv_path = params.get('mapping_class_csv_path', 'c_data_analysis_results')
    save_classification_prefix = params.get('save_classification_prefix', 'c_spike_rate_classification')
    prominence_C = params.get('prominence_C', 4.0)
    distance_C = params.get('distance_C', 5)
    histogram_bins = params.get('histogram_bins', 50)
    max_classification_heatmaps = params.get('max_classification_heatmaps', 100)
    enable_normalization = params.get('enable_normalization', True)
    plot_peak_stats_comparison = params.get('plot_peak_stats_comparison', True)
    plot_classification_heatmaps_flag = params.get('plot_classification_heatmaps_flag', True)
    plot_overall_peak_histograms = params.get('plot_overall_peak_histograms', True)
    save_peak_stats_csv = params.get('save_peak_stats_csv', True)
    save_classification_csv = params.get('save_classification_csv', True)
    save_individual_mapping_plots = params.get('save_individual_mapping_plots', False)
    individual_plots_output_base_dir = params.get('individual_plots_output_base_dir', 'individual_mapping_plots')

    original_id_col = 'session'
    aligned_id_col = 'session.1'

    peak_counts_results = []
    class_A_rows = []
    class_B_rows = []
    heatmap_data_A1, heatmap_data_A2 = [], []
    heatmap_data_B1, heatmap_data_B2 = [], []
    original_mapping_indices_A = []
    original_mapping_indices_B = []
    all_peak_magnitudes1 = []
    all_peak_magnitudes2 = []

    # 新增：用于保存峰值数据的列表
    spike_data_session1 = []
    spike_data_session2 = []

    start_row_index = 1 if not mappings_df.empty and mappings_df.iloc[0, 0] == 'description' else 0

    for i, row in mappings_df.iloc[start_row_index:].iterrows():
        try:
            cell1_id = int(float(row[original_id_col]))
            cell2_id = int(float(row[aligned_id_col]))
        except ValueError:
            print(f"Warning: Skipping mapping {i} due to non-numeric value in column '{original_id_col}' or '{aligned_id_col}'.")
            continue

        c_trace1 = ds1['C'].sel(unit_id=cell1_id).values
        c_trace2 = ds2['C'].sel(unit_id=cell2_id).values

        frames1 = len(c_trace1)
        frames2 = len(c_trace2)

        c_trace1_processed = c_trace1
        c_trace2_processed = c_trace2

        if enable_normalization:
            std_val1 = 1.0
            if std_df1 is not None and 'unit_id' in std_df1.columns and 'YRA_C_Std' in std_df1.columns:
                std_row1 = std_df1[std_df1['unit_id'] == cell1_id]['YRA_C_Std']
                if not std_row1.empty and std_row1.iloc[0] > 0:
                    std_val1 = std_row1.iloc[0]
            c_trace1_processed = c_trace1 / std_val1

            std_val2 = 1.0
            if std_df2 is not None and 'unit_id' in std_df2.columns and 'YRA_C_Std' in std_df2.columns:
                std_row2 = std_df2[std_df2['unit_id'] == cell2_id]['YRA_C_Std']
                if not std_row2.empty and std_row2.iloc[0] > 0:
                    std_val2 = std_row2.iloc[0]
            c_trace2_processed = c_trace2 / std_val2

        s1_spike_count, s1_peak_indices, s1_peak_values, s1_max_val, s1_mean_val, s1_std_val, s1_spike_rate = \
            _get_spike_stats(c_trace1_processed, prominence_C, distance_C, frames1)
        s2_spike_count, s2_peak_indices, s2_peak_values, s2_max_val, s2_mean_val, s2_std_val, s2_spike_rate = \
            _get_spike_stats(c_trace2_processed, prominence_C, distance_C, frames2)

        peak_counts_results.append({
            'Mapping_Index': i,
            f'{session1_name}_Unit_ID': cell1_id,
            f'{session2_name}_Unit_ID': cell2_id,
            f'{session1_name}_Peak_Count': s1_spike_count,
            f'{session2_name}_Peak_Count': s2_spike_count,
            f'{session1_name}_Peak_Max_Magnitude': s1_max_val,
            f'{session2_name}_Peak_Max_Magnitude': s2_max_val,
            f'{session1_name}_Peak_Mean_Magnitude': s1_mean_val,
            f'{session2_name}_Peak_Mean_Magnitude': s2_mean_val,
            f'{session1_name}_Peak_Std_Magnitude': s1_std_val,
            f'{session2_name}_Peak_Std_Magnitude': s2_std_val,
            f'{session1_name}_Spike_Rate_per_100_frames': s1_spike_rate,
            f'{session2_name}_Spike_Rate_per_100_frames': s2_spike_rate
        })

        # 新增：保存峰值数据
        spike_data_session1.append({
            'Unit_ID': cell1_id,
            'Peak_Indices': s1_peak_indices,
            'Peak_Values': s1_peak_values
        })
        spike_data_session2.append({
            'Unit_ID': cell2_id,
            'Peak_Indices': s2_peak_indices,
            'Peak_Values': s2_peak_values
        })

        min_frames_for_plotting = max(len(c_trace1_processed), len(c_trace2_processed))
        padded_trace1 = np.pad(c_trace1_processed, (0, max(0, min_frames_for_plotting - len(c_trace1_processed))), 'constant')
        padded_trace2 = np.pad(c_trace2_processed, (0, max(0, min_frames_for_plotting - len(c_trace2_processed))), 'constant')

        if s1_spike_rate > s2_spike_rate:
            class_A_rows.append(row)
            heatmap_data_A1.append(padded_trace1)
            heatmap_data_A2.append(padded_trace2)
            original_mapping_indices_A.append(i)
        else:
            class_B_rows.append(row)
            heatmap_data_B1.append(padded_trace1)
            heatmap_data_B2.append(padded_trace2)
            original_mapping_indices_B.append(i)

        all_peak_magnitudes1.extend(s1_peak_values)
        all_peak_magnitudes2.extend(s2_peak_values)

    if not peak_counts_results:
        print("No valid mapping data available for analysis.")
        return

    if save_peak_stats_csv:
        print("\n--- Saving C Trace Peak Statistics Data ---")
        results_df = pd.DataFrame(peak_counts_results)
        output_dir = os.path.dirname(peak_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results_df.to_csv(peak_csv_path, index=False)
        print(f"✔️ C trace peak statistics saved to: {peak_csv_path}")

    if save_classification_csv:
        print("\n--- Saving Mapped Pairs Classified by Spike Rate Difference ---")
        mapping_class_output_dir = os.path.dirname(mapping_class_csv_path) if os.path.isfile(mapping_class_csv_path) else mapping_class_csv_path
        os.makedirs(mapping_class_output_dir, exist_ok=True)
        pd.DataFrame(class_A_rows).to_csv(os.path.join(mapping_class_output_dir, f'{save_classification_prefix}_classA.csv'), index=False)
        pd.DataFrame(class_B_rows).to_csv(os.path.join(mapping_class_output_dir, f'{save_classification_prefix}_classB.csv'), index=False)
        print(f"✔️ Classified {len(class_A_rows)} mappings into Class A ({session1_name} spike rate > {session2_name} spike rate), and {len(class_B_rows)} mappings into Class B ({session2_name} spike rate ≥ {session1_name} spike rate).")
        print(f"✔️ Saved as {os.path.join(mapping_class_output_dir, f'{save_classification_prefix}_classA.csv')} and ..._classB.csv")

    # 新增：保存峰值数据到CSV文件
    spike_data_session1_df = pd.DataFrame(spike_data_session1)
    spike_data_session2_df = pd.DataFrame(spike_data_session2)
    spike_data_session1_df.to_csv(os.path.join(mapping_class_csv_path,'spike_data_session1.csv'), index=False)
    spike_data_session2_df.to_csv(os.path.join(mapping_class_csv_path,'spike_data_session2.csv'), index=False)
    print(f"✔️ Spike data for Session 1 saved to: {os.path.join(mapping_class_csv_path,'spike_data_session1.csv')}")
    print(f"✔️ Spike data for Session 2 saved to: {os.path.join(mapping_class_csv_path,'spike_data_session2.csv')}")

    if plot_peak_stats_comparison:
        print("\n--- Plotting C Matrix Statistics Comparison ---")
        stat_names = ['Peak_Count', 'Peak_Mean_Magnitude', 'Peak_Std_Magnitude', 'Spike_Rate_per_100_frames']
        display_names = ['Spike Count', 'Peak Mean Magnitude', 'Peak Std Magnitude', 'Spike Rate (%)']

        results_df = pd.DataFrame(peak_counts_results)

        num_plots = len(stat_names)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3), squeeze=False)

        for j, stat_name_raw in enumerate(stat_names):
            C1_vals = results_df[f'{session1_name}_{stat_name_raw}'].tolist()
            C2_vals = results_df[f'{session2_name}_{stat_name_raw}'].tolist()
            mapping_indices = results_df['Mapping_Index'].tolist()

            ax = axs[j, 0]
            ax.plot(mapping_indices, C1_vals, label=f'{session1_name} - C', color='blue', marker='o', markersize=4)
            ax.plot(mapping_indices, C2_vals, label=f'{session2_name} - C', color='red', marker='x', markersize=4)
            ax.set_title(f"C Matrix - {display_names[j]}")
            ax.set_xlabel('Mapping Index')
            ax.set_ylabel(display_names[j])
            ax.grid(True)
            ax.legend()

        plt.tight_layout()

        picture_output_dir = os.path.join(mapping_class_csv_path, 'pictures')
        os.makedirs(picture_output_dir, exist_ok=True)

        fig.savefig(os.path.join(picture_output_dir, 'C_matrix_stats_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"C matrix statistics comparison plot saved to: {os.path.join(picture_output_dir, 'C_matrix_stats_comparison.png')}")
        plt.show()

    if plot_classification_heatmaps_flag:
        print("\n--- Plotting Classification Heatmaps (C Matrix) ---")

        def plot_classification_heatmaps(data1, data2, title_suffix, s1_name, s2_name, y_labels, label):
            if not data1 or not data2:
                print(f"⚠️ No data available for plotting heatmap: {title_suffix}")
                return

            max_trace_len = max(max(len(t) for t in data1), max(len(t) for t in data2)) if data1 and data2 else 0
            data1_padded = [np.pad(t, (0, max_trace_len - len(t)), 'constant') for t in data1]
            data2_padded = [np.pad(t, (0, max_trace_len - len(t)), 'constant') for t in data2]

            data1_arr = np.array(data1_padded)
            data2_arr = np.array(data2_padded)

            vmax = max(data1_arr.max(), data2_arr.max()) if data1_arr.size > 0 and data2_arr.size > 0 else 1
            vmax = max(vmax, 0.01)

            fig_heatmap = plt.figure(figsize=(12, max(5, len(data1_arr) / 20 * 2)))
            gs = fig_heatmap.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)

            ax0 = fig_heatmap.add_subplot(gs[0, 0])
            ax1 = fig_heatmap.add_subplot(gs[0, 1], sharey=ax0)

            cbar_ax = fig_heatmap.add_axes([0.95, 0.15, 0.02, 0.7])

            im0 = ax0.imshow(data1_arr, aspect='auto', cmap='viridis', vmax=vmax, origin='lower')
            ax0.set_title(f'{s1_name} C Matrix ({title_suffix})')
            ax0.set_ylabel('Mapping Index')
            ax0.set_xlabel('Frame')
            ax0.grid(False)

            im1 = ax1.imshow(data2_arr, aspect='auto', cmap='viridis', vmax=vmax, origin='lower')
            ax1.set_title(f'{s2_name} C Matrix ({title_suffix})')
            ax1.set_xlabel('Frame')
            ax1.grid(False)

            num_displayed_mappings = len(y_labels)
            tick_interval = max(1, num_displayed_mappings // 10)
            tick_locations = np.arange(0, num_displayed_mappings, tick_interval)
            tick_labels = [y_labels[int(loc)] for loc in tick_locations]

            ax0.set_yticks(tick_locations)
            ax0.set_yticklabels(tick_labels)

            plt.setp(ax1.get_yticklabels(), visible=False)

            fig_heatmap.colorbar(im0, cax=cbar_ax, label='Signal Intensity' + (' (STD Normalized)' if enable_normalization else ''))

            plt.suptitle(f'{title_suffix}: C Matrix Signal Heatmap', fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

            picture_output_dir = os.path.join(mapping_class_csv_path, 'pictures')
            os.makedirs(picture_output_dir, exist_ok=True)

            fig_heatmap.savefig(os.path.join(picture_output_dir, f'heatmap_{label}.png'), dpi=300, bbox_inches='tight')
            print(f"{title_suffix} heatmap saved to: {os.path.join(picture_output_dir, f'{title_suffix}_heatmap.png')}")
            plt.show()

        plot_classification_heatmaps(heatmap_data_A1, heatmap_data_A2, 
                                     f'Class A ({session1_name} spike rate > {session2_name} spike rate)', 
                                     session1_name, session2_name, original_mapping_indices_A,1)
        plot_classification_heatmaps(heatmap_data_B1, heatmap_data_B2, 
                                     f'Class B ({session2_name} spike rate ≥ {session1_name} spike rate)', 
                                     session1_name, session2_name, original_mapping_indices_B,2)

    if plot_overall_peak_histograms:
        print("\n--- Plotting C Matrix Peak Magnitude Distribution Histograms for All Mapped Pairs ---")
        fig_hist_overall, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

        magnitude_label = 'Peak Magnitude' + (' (STD Normalized)' if enable_normalization else '')

        if len(all_peak_magnitudes1) > 0:
            axs[0].hist(all_peak_magnitudes1, bins=histogram_bins, color='blue', alpha=0.7, 
                         label=f'{session1_name} (N={len(all_peak_magnitudes1)} peaks)')
            axs[0].set_title(f'C Peak Magnitude Distribution for All Cells in {session1_name}')
            axs[0].set_xlabel(magnitude_label)
            axs[0].set_ylabel('Frequency')
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].set_title(f'No Peaks for {session1_name}')
            axs[0].set_xlabel(magnitude_label)
            axs[0].set_ylabel('Frequency')

        if len(all_peak_magnitudes2) > 0:
            axs[1].hist(all_peak_magnitudes2, bins=histogram_bins, color='red', alpha=0.7, 
                         label=f'{session2_name} (N={len(all_peak_magnitudes2)} peaks)')
            axs[1].set_title(f'C Peak Magnitude Distribution for All Cells in {session2_name}')
            axs[1].set_xlabel(magnitude_label)
            axs[1].set_ylabel('Frequency')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_title(f'No Peaks for {session2_name}')
            axs[1].set_xlabel(magnitude_label)
            axs[1].set_ylabel('Frequency')

        plt.tight_layout()

        picture_output_dir = os.path.join(mapping_class_csv_path, 'pictures')
        os.makedirs(picture_output_dir, exist_ok=True)

        fig_hist_overall.savefig(os.path.join(picture_output_dir, 'Overall_C_matrix_peak_magnitude_distribution_histograms.png'), dpi=300, bbox_inches='tight')
        print(f"Overall C matrix peak magnitude distribution histograms saved to: {os.path.join(picture_output_dir, 'Overall_C_matrix_peak_magnitude_distribution_histograms.png')}")
        plt.show()

    if save_individual_mapping_plots:
        print("\n--- Saving individual mapping plots in batch mode ---")
        os.makedirs(individual_plots_output_base_dir, exist_ok=True)

        for i, row in mappings_df.iloc[start_row_index:].iterrows():
            try:
                cell1_id_map = int(float(row[original_id_col]))
                cell2_id_map = int(float(row[aligned_id_col]))
            except ValueError:
                print(f"Warning: Skipping mapping {i} due to non-numeric value in column '{original_id_col}' or '{aligned_id_col}'.")
                continue

            mapping_pair_folder = os.path.join(individual_plots_output_base_dir, f'M{i}-S1-{cell1_id_map}-S2-{cell2_id_map}')
            os.makedirs(mapping_pair_folder, exist_ok=True)
            print(f"  Processing and saving plots for mapping pair: M{i}-S1-{cell1_id_map}-S2-{cell2_id_map} to {mapping_pair_folder}")

            _plot_minian_C_traces_and_mappings(
                cell1_id=cell1_id_map,
                cell2_id=cell2_id_map,
                show_overlayed_traces=True,
                show_individual_traces=True,
                detect_spikes=True,
                spike_prominence=prominence_C,
                spike_distance=distance_C,
                enable_normalization=enable_normalization,
                show_peak_distribution=True,
                histogram_bins=histogram_bins,
                show_full_fov=True,
                show_local_view=True,
                show_cell1_individual_map=True,
                show_cell2_individual_map=True,
                show_overlap_map_local=True,
                local_padding=params.get('local_padding', 80),
                min_crop_size=params.get('min_crop_size', 150),
                save_path_prefix=mapping_pair_folder,
                mapping_idx=i
            )

    print("\n--- Comprehensive C Matrix Analysis and Plotting Complete ---")
# --- Main Interactive Analyzer (Responsible for instantiating the global data manager) ---
def interactive_minian_analyzer(
    mappings_csv_path: str,
    session1_nc_path: str,
    session2_nc_path: str,
    std_csv_path1: str = None,
    std_csv_path2: str = None,
    cents_csv_path: str = None,
    # Unified parameter dictionary for C matrix analysis
    c_analysis_params: dict = None
):
    global _global_minian_data_manager
    _global_minian_data_manager = MinianDataManager(
        mappings_csv_path=mappings_csv_path,
        session1_nc_path=session1_nc_path,
        session2_nc_path=session2_nc_path,
        std_csv_path1=std_csv_path1,
        std_csv_path2=std_csv_path2,
        cents_csv_path=cents_csv_path
    )
    
    data = _global_minian_data_manager.get_data()
    if data["mappings_df"] is None:
        print("Error: Mapping data could not be loaded. Please check the provided file paths and try again.")
        return

    # Set default parameters for C matrix analysis and mapping visualization
    default_c_analysis_params = {
        'peak_csv_path': 'c_trace_peak_stats.csv',
        'mapping_class_csv_path': 'c_data_analysis_results', # Changed default to a folder name
        'save_classification_prefix': 'c_spike_rate_classification',
        'prominence_C': 4.0,
        'distance_C': 5,
        'histogram_bins': 50,
        'max_classification_heatmaps': 100,
        'enable_normalization': True, # Default to enable normalization
        'plot_peak_stats_comparison': True, # Default to plot peak statistics comparison
        'plot_classification_heatmaps_flag': True, # Default to plot classification heatmaps
        'plot_overall_peak_histograms': True, # Default to plot overall peak histograms
        'save_peak_stats_csv': True, # Default to save peak statistics CSV
        'save_classification_csv': True, # Default to save classification CSV
        'show_selected_c_traces': True, # Default to show C traces for selected unit
        'show_selected_mapping_full_fov': True, # Default to show full FOV map for selected unit
        'show_selected_mapping_local_view': True, # Default to show local view map for selected unit
        'show_selected_cell1_individual_map': True, # Default to show individual map for cell 1
        'show_selected_cell2_individual_map': True, # Default to show individual map for cell 2
        'local_padding': 80, # Pixels around centroid for local view
        'min_crop_size': 150, # Minimum size for local view crop
        'save_individual_mapping_plots': True, # NEW: Default to False
        'individual_plots_output_base_dir': 'individual_mapping_plots' # NEW: Default output dir
    }
    # Merge user-provided parameters, overriding defaults
    c_analysis_config = default_c_analysis_params.copy()
    if c_analysis_params is not None:
        c_analysis_config.update(c_analysis_params)


    mappings_df = data["mappings_df"]
    # Adjusting for the header in mappings.csv if it exists
    original_id_col = 'session'
    aligned_id_col = 'session.1'
    
    mapping_options = []
    # Determine the starting row for iteration based on whether the first row is a descriptive header
    start_row_idx = 1 if not mappings_df.empty and mappings_df.iloc[0, 0] == 'description' else 0

    if not mappings_df.empty and len(mappings_df) > start_row_idx:
        for i, row in mappings_df.iloc[start_row_idx:].iterrows():
            if pd.notna(row[original_id_col]) and pd.notna(row[aligned_id_col]):
                try:
                    s1_id = int(float(row[original_id_col]))
                    s2_id = int(float(row[aligned_id_col]))
                    option_label = f"Mapping Pair {i}: S1 ID {s1_id} <-> S2 ID {s2_id}"
                    mapping_options.append((option_label, (s1_id, s2_id)))
                except ValueError:
                    continue
    if not mapping_options:
        mapping_options.append(("No valid pairs found", (None, None)))
        print("Warning: No valid mapping pairs found in mappings.csv.")

    global_selected_cell1_id = None
    global_selected_cell2_id = None

    mapping_selector = widgets.Dropdown(
        options=mapping_options,
        value=mapping_options[0][1] if mapping_options and mapping_options[0][1][0] is not None else (None, None),
        description='Select Mapping Pair:',
    )
    
    plot_selected_pair_button = widgets.Button(description="Plot Selected Mapping Pair & C Traces")
    run_full_c_analysis_button = widgets.Button(description="Run Comprehensive C Matrix Analysis (All Pairs)")
    output_area = widgets.Output()

    def on_selection_change(change):
        nonlocal global_selected_cell1_id, global_selected_cell2_id
        selected_ids = change['new']
        global_selected_cell1_id = selected_ids[0]
        global_selected_cell2_id = selected_ids[1]
        with output_area:
            clear_output(wait=True)
            if global_selected_cell1_id is not None:
                print(f"Selected mapping pair: Session 1 ID {global_selected_cell1_id} <-> Session 2 ID {global_selected_cell2_id}")
            else:
                print("Please select a valid mapping pair.")

    def on_plot_selected_pair_button_click(_):
        with output_area:
            clear_output(wait=True)
            if global_selected_cell1_id is None:
                print("Please select a mapping pair first.")
                return
            
            _plot_minian_C_traces_and_mappings(
                cell1_id=global_selected_cell1_id,
                cell2_id=global_selected_cell2_id,
                show_overlayed_traces=c_analysis_config['show_selected_c_traces'],
                show_individual_traces=c_analysis_config['show_selected_c_traces'],
                detect_spikes=True, # Always detect spikes for individual plots for full info
                spike_prominence=c_analysis_config['prominence_C'],
                spike_distance=c_analysis_config['distance_C'],
                enable_normalization=c_analysis_config['enable_normalization'],
                show_peak_distribution=c_analysis_config['plot_overall_peak_histograms'], # Reuse this flag for individual peak dist
                histogram_bins=c_analysis_config['histogram_bins'],
                # Mapping visualization parameters
                show_full_fov=c_analysis_config['show_selected_mapping_full_fov'],
                show_local_view=c_analysis_config['show_selected_mapping_local_view'],
                show_cell1_individual_map=c_analysis_config['show_selected_cell1_individual_map'],
                show_cell2_individual_map=c_analysis_config['show_selected_cell2_individual_map'],
                show_overlap_map_local=True, # Always show overlap for interactive selected plot
                local_padding=c_analysis_config['local_padding'],
                min_crop_size=c_analysis_config['min_crop_size'],
                save_path_prefix=None, # In interactive mode, we don't save
                mapping_idx=None # Not applicable for single interactive plot
            )
            
    def on_run_full_c_analysis_button_click(_):
        with output_area:
            clear_output(wait=True)
            print("\n--- Running Comprehensive C Matrix Analysis and Plotting (All Mapped Pairs) ---")
            analyze_and_plot_c_data(params=c_analysis_config)
            print("\n--- Comprehensive C Matrix Analysis and Plotting Complete. ---")

    # Bind events
    mapping_selector.observe(on_selection_change, names='value')
    plot_selected_pair_button.on_click(on_plot_selected_pair_button_click)
    run_full_c_analysis_button.on_click(on_run_full_c_analysis_button_click)

    # Initial display
    display(widgets.VBox([mapping_selector, plot_selected_pair_button, run_full_c_analysis_button]), output_area)

    # Automatically select the first option and display info on first load
    if mapping_selector.value[0] is not None:
        on_selection_change({'new': mapping_selector.value})    