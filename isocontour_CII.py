# this code is used to create the 3D plot of the CII line distribution

from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import plotly.graph_objects as go
import numpy as np

vmin = 0
vmax = 12

filename = "../FEEDBACK_W40_CII_OC9N.lmv.fits"
# filename = "cii_in_galac.fits"
# filename = "w40_c18o.fits"
u.add_enabled_units(u.def_unit(["K (Tmb)"], represents=u.K))
# u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
cube = SpectralCube.read(filename)
cube_slab = cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)

import numpy as np
from matplotlib.path import Path

# Define the polygon
polygon = Path([(72, 240), (231, 173), (173, 15), (15, 76)])

# Create a grid of the same shape as the cube
y, x = np.mgrid[: cube_slab.shape[1], : cube_slab.shape[2]]

# Create a mask of the same shape as the cube, with True inside the polygon and False outside
mask = polygon.contains_points(np.vstack((x.flatten(), y.flatten())).T).reshape(
    cube_slab.shape[1], cube_slab.shape[2]
)

cube_slab_quantity = cube_slab.unmasked_data[:, :, :]

# Apply the mask to each plane of the cube
for i in range(cube_slab.shape[0]):
    cube_slab_quantity[i, ~mask] = np.nan

# Convert the Quantity back to a SpectralCube
cube_slab = SpectralCube(
    data=cube_slab_quantity,
    wcs=cube_slab.wcs,
    mask=cube_slab.mask,
    meta=cube_slab.meta,
    header=cube_slab.header,
)

from scipy.ndimage import convolve
from astropy import units as u

# Define the kernel
n = 5
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
x = X.flatten()[::2]
y = Y.flatten()[::2]
z = Z.flatten()[::2]
transposed_data = np.transpose(cube_slab.hdu.data, (2, 1, 0))
value = transposed_data.flatten()[::2]
min_value = 15

value[value < min_value] = np.nan
mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(value)

# Apply the masks to the data
x_valid = x[mask]
y_valid = y[mask]
z_valid = z[mask]
value_valid = value[mask]

import numpy as np
import plotly.graph_objects as go

fig = go.Figure(
    data=go.Scatter3d(
        x=x_valid,
        y=y_valid,
        z=z_valid,
        mode="markers",
        marker=dict(
            size=8,
            color=value_valid,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.05,
            colorbar=dict(title="K (Tmb)"),  # add a colorbar with the title 'Values'
            symbol='square'
        ),
    )
)
tickvals = np.linspace(lon1, lon2, num=10)  # adjust as needed
ticktext = [str(round(lon2 - val + lon1, 2)) for val in tickvals]
print(ticktext)
fig.update_layout(
    scene=dict(
        # xaxis_title="Galactic Longitude (degrees)",
        xaxis_title="RA (degrees)",
        yaxis_title="Dec (degrees)",
        zaxis_title="Velocity (km/s)",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),

    ),
    title="CII 158 μm",
)

fig.show()
import plotly.io as pio

# ... your figure creation code ...

pio.write_html(fig, "sofiaCII.html")
