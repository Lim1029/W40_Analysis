# this code is used to create the 3D plot of the molecular line distribution

from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import convolve
from astropy import units as u

vmin = 0
vmax = 12

files = [
    ("../fits_images/fits_files/w40_c18o_1-9.fits", "Viridis", 0.5, "C18O", 0.99),
    ("../fits_images/fits_files/fixed_w40_c2h_1-8.fits", "Hot", 0.175, "C2H", 0.95),
    ("../fits_images/fits_files/fixed_w40_hcn_1-10.fits", "Jet", 0.3, "HCN", 0.9),
    ("../fits_images/fits_files/fixed_w40_n2hp_1-8.fits", "Cividis", 0.15, "N2H+", 0.85),
    ("../fits_images/fits_files/w40_hcop_1-10.fits", "Inferno", 0.3, "HCO+", 0.8),
    ("../fits_images/fits_files/cn_modified_2.fits", "Magma", 0.4, "CN", 0.75),
]
fig = go.Figure()
for i, (filename, colorscheme, valuemin, linename, colorbar_pos) in enumerate(files):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    cube = SpectralCube.read(filename)
    cube_slab = cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)

    # Define the kernel
    n = 50
    kernel = np.array(
        [
            [[n, n, n], [n, n, n], [n, n, n]],
            [[n, n, n], [n, 0, n], [n, n, n]],
            [[n, n, n], [n, n, n], [n, n, n]],
        ]
    )

    # Convert the SpectralCube to a Quantity
    cube_slab_quantity = cube_slab.unmasked_data[:, :, :].value

    # Apply the convolution
    convolved_data = convolve(cube_slab_quantity, kernel, mode="constant", cval=0.0)

    # Calculate the average
    average_data = convolved_data / np.sum(kernel)

    # Convert the Quantity back to a SpectralCube
    cube_slab = SpectralCube(
        data=average_data * u.K,
        wcs=cube_slab.wcs,
        mask=cube_slab.mask,
        meta=cube_slab.meta,
        header=cube_slab.header,
    )

    # cube_slab = cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    lon1 = cube_slab.longitude_extrema.value[0]
    lat1 = cube_slab.latitude_extrema.value[0]
    lon2 = cube_slab.longitude_extrema.value[1]
    lat2 = cube_slab.latitude_extrema.value[1]
    # print(lon1, lon2, lat1, lat2)
    # Create a 3D grid of x, y, z positions
    x = np.arange(cube_slab.shape[2])
    y = np.arange(cube_slab.shape[1])
    z = np.arange(cube_slab.shape[0])
    z = np.linspace(vmin, vmax, cube_slab.shape[0])
    x = np.linspace(
        lon1,
        lon2,
        cube_slab.shape[2],
    )
    y = np.linspace(
        lat1,
        lat2,
        cube_slab.shape[1],
    )
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    transposed_data = np.transpose(cube_slab.hdu.data, (2, 1, 0))
    value = transposed_data.flatten()
    min_value = valuemin

    value[value < min_value] = np.nan

    # ################################
    from scipy.ndimage import generate_binary_structure, binary_dilation

    radius = 5
    value_reshaped = value.reshape(cube_slab.shape)

    # Initialize a mask for the pixels to be set to nan
    nan_mask = np.zeros_like(value_reshaped, dtype=bool)

    # Iterate over each pixel in the value array
    for index, pixel in np.ndenumerate(value_reshaped):
        # Skip if the pixel is already nan
        if np.isnan(pixel):
            continue

        # Define a slice for the neighborhood of the pixel
        slices = tuple(
            slice(max(0, i - radius), min(value_reshaped.shape[j], i + radius + 1))
            for j, i in enumerate(index)
        )

        # Get the neighborhood of the pixel
        neighborhood = value_reshaped[slices]

        # Check the number of non-nan values in the neighborhood
        non_nan_count = np.count_nonzero(~np.isnan(neighborhood))

        # If the number of non-nan values is less than 10, mark the pixel to be set to nan
        if non_nan_count < 20:
            nan_mask[index] = True

    # Set the marked pixels to nan
    value_reshaped[nan_mask] = np.nan

    value = value_reshaped.flatten()
    # ################################

    # Add the code here
    ################################
    # from scipy.ndimage import convolve

    # # Define a 3D kernel that checks for the presence of valid neighbors
    # kernel = np.ones((3, 3, 3))
    # kernel[1, 1, 1] = 0

    # # Reshape the value array back to its original shape
    # value_reshaped = value.reshape(transposed_data.shape)

    # # Convolve the valid values with the kernel
    # convolved = convolve(~np.isnan(value_reshaped), kernel)

    # # Create a mask for the isolated pixels
    # isolated_mask = convolved == 0

    # # Set isolated pixels to nan
    # value_reshaped[isolated_mask] = np.nan

    # # Flatten the value array again
    # value = value_reshaped.flatten()
    ################################

    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(value)

    # Apply the masks to the data
    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = z[mask]
    value_valid = value[mask]

    import numpy as np
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter3d(
            visible="legendonly",
            x=x_valid,
            y=y_valid,
            z=z_valid,
            mode="markers",
            marker=dict(
                size=8,
                color=value_valid,  # set color to an array/list of desired values
                colorscale=colorscheme,  # choose a colorscale
                opacity=0.05,
                colorbar=dict(
                    title="K (Ta*)",
                    x=colorbar_pos,  # Adjust the x-position of the colorbar
                    # len=0.7,  # Adjust the length of the colorbar
                ),  # add a colorbar with the title 'Values'
                symbol="square",
            ),
            name=linename,  # set the name of the trace to linename
            showlegend=True,
        )
    )

fig.data[0].visible = True

tickvals = np.linspace(lon1, lon2, num=10)  # adjust as needed
ticktext = [str(round(lon2 - val + lon1, 2)) for val in tickvals]
# print(ticktext)
fig.update_layout(
    scene=dict(
        # xaxis_title="Galactic Longitude (degrees)",
        xaxis_title="Galactic Longitude (degrees)",
        yaxis_title="Galactic Latitude (degrees)",
        zaxis_title="Velocity (km/s)",
        xaxis=dict(
            tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[lon1, lon2]
        ),
        yaxis=dict(range=[lat1, lat2]),
        zaxis=dict(range=[vmin, vmax]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    ),
    title="W40 Molecular Line Distribution",
    title_x=0.5,
    legend=dict(
        font=dict(
            size=30,  # adjust as needed
        ),
        # entrywidth=1,
        itemsizing="constant",
        itemwidth=50,
    ),
)

#################################
# Create a list of buttons for the updatemenus
# buttons = []
# for i, (filename, colorscheme, valuemin, linename) in enumerate(files):
#     visibility = [False] * len(files)
#     visibility[i] = True
#     buttons.append(
#         dict(
#             label=linename,  # use linename as the label
#             method="update",
#             args=[{"visible": visibility}, {"title": linename}],
#         )  # use linename as the title
#     )

# Add a button for showing all traces
# buttons.append(
#     dict(
#         label="All",
#         method="update",
#         args=[{"visible": [True] * len(files)}, {"title": "All"}],
#     )
# )

# Add the updatemenus to the layout
# fig.update_layout(
#     updatemenus=[
#         dict(
#             type="dropdown", direction="down", active=0, x=0.57, y=1.2, buttons=buttons
#         )
#     ]
# )
#################################

fig.show()
import plotly.io as pio

# ... your figure creation code ...

pio.write_html(fig, "w40_molecular.html")
