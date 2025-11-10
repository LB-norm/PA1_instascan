import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_hyperparam_parallel_coordinates(data, fitness_col=None, 
                                         color_scale='Viridis', 
                                         title="YOLOv11 Segmentation, batchsize=8, modelsize=large"):
    """
    Reads hyperparameter combinations + fitness scores and plots
    a plotly parallel-coordinates figure.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to CSV file or a DataFrame. The first column is assumed
        to be the fitness score unless `fitness_col` is specified.
    fitness_col : str, optional
        Name of the column to use for coloring. Defaults to the first column.
    color_scale : str or list, optional
        Plotly continuous color scale name or list of colors.
    title : str, optional
        Title of the figure.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object. Call `fig.show()` to render.
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("`data` must be a path to CSV or a pandas DataFrame.")

    # Determine fitness column
    if fitness_col is None:
        fitness_col = df.columns[0]

    # Build dimensions list: omit fitness column
    dimensions = [c for c in df.columns if c != fitness_col] + [fitness_col]

    # Create the figure
    fig = px.parallel_coordinates(
        df,
        color=fitness_col,
        dimensions=dimensions,
        color_continuous_scale=color_scale,
        title=title
    )

    # Tweak layout for readability
    fig.update_layout(
        coloraxis_colorbar=dict(title=fitness_col),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def plot_parallel_with_fitness(
    df,
    fitness_col,
    color_scale='Viridis',
    title="Parallel Coordinates (with Fitness)"
):
    # build your axis order
    dims = [c for c in df.columns if c != fitness_col] + [fitness_col]

    # create a Parcoords trace with explicit line width
    trace = go.Parcoords(
        line=dict(
            color=df[fitness_col],
            colorscale=color_scale,
            showscale=True,
        ),
                dimensions=[
            dict(
                label=c,
                values=df[c],
                tickfont=dict(size=22),
                labelfont=dict(size=22)
            )
            for c in dims
        ]
        # dimensions=[{'label': c, 'values': df[c]} for c in dims]
    )

    fig = go.Figure(trace)
    fig.update_layout(
        title=title,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# df = pd.read_csv(r"C:\Python_Projects\PA1 SKALA YOLO\best_HP\seg_8_Batch\tune4\tune_results.csv")
# columns1 = ['fitness', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
#        'warmup_momentum', 'box', 'cls']
# columns2 = ['fitness', 'dfl', 'hsv_h', 'hsv_s', 'hsv_v',
#        'translate', 'scale', 'fliplr', 'mosaic']

# df_cropped1 = df[columns1]
# df_cropped2 = df[columns2]

# fig1 = plot_hyperparam_parallel_coordinates(df_cropped1)
# fig2 = plot_hyperparam_parallel_coordinates(df_cropped2)

# fig1.show()
# fig2.show()

def plot_image_grid(folder_path, target_size=(350, 300)):
    """
    Reads exactly 10 images from `folder_path`, resizes them to `target_size`,
    and plots them in a 2x5 grid.
    
    Parameters:
    - folder_path (str): Path to the folder containing exactly 10 images.
    - target_size (tuple of int): Desired (width, height) in pixels for each image.
    
    Raises:
    - ValueError: If the folder does not contain exactly 10 images.
    """
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # List and validate image files
    files = [f for f in os.listdir(folder_path) 
             if f.lower().endswith(image_extensions)]
    if len(files) != 10:
        raise ValueError(f"Expected exactly 10 images in '{folder_path}', found {len(files)}.")
    
    files.sort()  # Optional: sort for consistent ordering
    
    # Determine appropriate resampling filter for this Pillow version
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    
    # Load and resize images
    images = []
    for fname in files:
        path = os.path.join(folder_path, fname)
        img = Image.open(path)
        img = img.resize(target_size, resample_filter)
        images.append(img)
    
    # Create the plot
    fig, axes = plt.subplots(2, 5, 
                             figsize=(5 * target_size[0] / 100, 
                                      2 * target_size[1] / 100))
    
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_iou_averages(iou_data):
    """
    Plots the average IoU for each model in descending order of IoU, with rotated x-axis labels.

    Parameters:
    - iou_data (dict): A dictionary where keys are model names (str)
      and values are lists of IoU values (float) for each test image.

    Example:
    >>> iou_data = {
    ...     'YOLOv3-very-long-model-name': [0.65, 0.70, 0.72, 0.68],
    ...     'YOLOv4-another-very-long-model-name': [0.75, 0.80, 0.78, 0.77],
    ...     'YOLOv5-yet-another-long-model-name': [0.82, 0.85, 0.83, 0.80]
    ... }
    >>> plot_iou_averages(iou_data)
    """
    # Compute average IoU for each model
    avg_iou = {model: (sum(vals) / len(vals) if vals else 0)
               for model, vals in iou_data.items()}

    # Sort models by average IoU descending
    sorted_models = sorted(avg_iou, key=avg_iou.get, reverse=True)
    sorted_ious = [avg_iou[model] for model in sorted_models]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_models, sorted_ious, color='skyblue', edgecolor='black')
    
    # Rotate the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add numerical labels above the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', 
                 ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Average IoU')
    plt.title('Average IoU per Model (Sorted)')
    plt.ylim(0.8, 1)
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_model_predictions(model_iou_dict, threshold):
    """
    Evaluates model predictions by filtering out IoU values below a threshold.
    
    Parameters:
        model_iou_dict (dict): Dictionary with keys as model names and values as lists of IoU scores.
        threshold (float): IoU threshold; predictions below this value are considered fails.
    
    Returns:
        filtered_predictions (dict): Dictionary with model names as keys and lists of IoU values 
                                     that are >= threshold.
        average_iou (dict): Dictionary mapping each model to its average IoU after filtering.
    """
    model_names = []
    averages = []
    fails = []
    filtered_predictions = {}
    
    # Evaluate each model
    for model, iou_list in model_iou_dict.items():
        model_names.append(model)
        
        # Filter predictions with IoU >= threshold
        good_predictions = [iou for iou in iou_list if iou >= threshold]
        filtered_predictions[model] = good_predictions
        
        # Count predictions below threshold (failures)
        fail_count = len([iou for iou in iou_list if iou < threshold])
        fails.append(fail_count)
        
        # Compute average IoU from the filtered predictions; handle empty list by setting NaN
        avg = np.mean(good_predictions) if good_predictions else np.nan
        averages.append(avg)

    # Zip the model names, averages, and good prediction counts together
    sorted_data = sorted(zip(model_names, averages, fails), key=lambda x: x[1], reverse=True)
    # Unzip the sorted data back into separate lists
    model_names, averages, fails = zip(*sorted_data)

    # Create a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_avg = 'tab:blue'

    # Rotate x-axis labels by 45 degrees
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average IoU', color=color_avg)
    ax1.plot(model_names, averages, marker='o', color=color_avg, label='Average IoU')
    ax1.tick_params(axis='y', labelcolor=color_avg)
    ax1.set_ylim(0.8, 1)
        # Annotate the average IoU for each model
    for i, avg in enumerate(averages):
        if not np.isnan(avg):
            ax1.annotate(f'{avg:.3}', 
                         xy=(model_names[i], avg), 
                         xytext=(0, 5),  # 5 pixels vertical offset
                         textcoords="offset points",
                         ha='center', 
                         va='bottom', 
                         color=color_avg)

    # Right axis: Number of fails
    ax2 = ax1.twinx()
    color_fail = 'tab:red'
    ax2.set_ylabel('Number of Fails (IoU < 0.9)', color=color_fail)
    bars = ax2.bar(model_names, fails, alpha=0.5, color=color_fail, label='Number of Fails (IoU < 0.9)')
    ax2.tick_params(axis='y', labelcolor=color_fail)

    # Annotate each bar with the fail count above it
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )

    # Title and layout
    plt.title('Model Evaluation: Average IoU for good detections and Number of Fails')
    fig.tight_layout()  # Ensures the plot fits nicely without overlap
    plt.show()
    
    # Return results for further analysis if needed
    average_iou = dict(zip(model_names, averages))
    fails = dict(zip(model_names, fails))
    return filtered_predictions, average_iou, fails