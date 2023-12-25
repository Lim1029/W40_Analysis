import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd
from astropy.convolution import convolve, Box1DKernel
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy import units as u
import numpy as np
from spectral_cube import SpectralCube
from astropy import units as u


def channel_to_velocity(channel, header):
    velo_lsr = header["VELO-LSR"] / 1000
    deltav = header["DELTAV"] / 1000
    crpix1 = header["CRPIX1"]
    return (channel - crpix1) * deltav + velo_lsr


def velocity_to_channel(velocity, header):
    velo_lsr = header["VELO-LSR"] / 1000
    deltav = header["DELTAV"] / 1000
    crpix1 = header["CRPIX1"]
    return int((velocity - velo_lsr) / deltav + crpix1)


lines = {
    "c2h": "w40_c2h_1-8.sdd",
    "c18o": "w40_c18o_1-9.sdd",
    "cn": "w40_cn_1-9.sdd",
    "hcn": "w40_hcn_1-10.sdd",
    "hcop": "w40_hcop_1-10.sdd",
    "n2hp": "w40_n2hp_1-8.sdd",
}

fits_name = {
    "c2h": "w40_c2h_1-8.fits",
    "c18o": "w40_c18o_1-9.fits",
    "cn": "w40_cn_1-9.fits",
    "hcn": "w40_hcn_1-10.fits",
    "hcop": "w40_hcop_1-10.fits",
    "n2hp": "w40_n2hp_1-8.fits",
}

cmap_dict = {
    "c2h": "Greys",
    "c18o": "Purples",
    "cn": "Blues",
    "hcn": "Greens",
    "hcop": "Oranges",
    "n2hp": "Reds",
}

color_dict = {
    "c2h": "grey",
    "c18o": "purple",
    "cn": "blue",
    "hcn": "green",
    "hcop": "orange",
    "n2hp": "red",
}

lines = ["c2h", "c18o", "cn", "hcn", "hcop", "n2hp"]

n_B_dict = {
    "c2h": 0.496,
    "cn": 0.399,
    "hcn": 0.496,
    "n2hp": 0.491,
}

v_dict = {
    "c2h": 93.173777e9,
    "cn": 113.49964430e9,
    "hcn": 88.63184750e9,
    "n2hp": 93.17376990e9,
}

B_0_dict = {
    "c2h": 43.67452e9,
    "cn": 56.69347e9,
    "hcn": 44.31597e9,
    "n2hp": 46.58687e9,
}

g_u_dict = {
    "c2h": 4,
    "cn": 4,
    "hcn": 3,
    "n2hp": 3,
}

A_ij_dict = {
    "c2h": 10 ** (-5.81605),
    "cn": 10 ** (-4.97352),
    "hcn": 10 ** (-4.61848),
    "n2hp": 10 ** (-4.44038),
}


def spitzer_on_galactic(axx, xlim=(0, 61), ylim=(0, 61), cmap="cubehelix", vmax=622,alpha=1):
    path_equitorial_down = "../30002561.30002561-33223-short.IRAC.4.median_mosaic.fits"
    path_equitorial_up = "../30002561.30002561-33944-short.IRAC.4.median_mosaic.fits"
    wcs_equitorial_down = WCS(fits.open(path_equitorial_down)[0].header)
    wcs_equitorial_up = WCS(fits.open(path_equitorial_up)[0].header)

    # path = 'w40_wise12_continuum.fits'
    a = axx.imshow(
        fits.open(path_equitorial_down)[0].data,
        origin="lower",
        cmap=cmap,
        vmin=28,
        vmax=vmax,
        transform=axx.get_transform(wcs_equitorial_down),
        alpha=alpha,
    )

    a = axx.imshow(
        fits.open(path_equitorial_up)[0].data,
        origin="lower",
        cmap=cmap,
        vmin=28,
        vmax=vmax,
        transform=axx.get_transform(wcs_equitorial_up),
        alpha=alpha,
    )
    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    axx.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    axx.set_ylim(ymin, ymax)
    axx.coords[0].set_axislabel(" ")
    axx.coords[1].set_axislabel(" ")


def spitzer_on_galactic2(
    axx, xlim=(0, 61), ylim=(0, 61), cmap="cubehelix", vmax=622, alpha=1
):
    path_equitorial_down = "../spitzer_mosaic.fits"
    wcs_equitorial_down = WCS(fits.open(path_equitorial_down)[0].header)

    # path = 'w40_wise12_continuum.fits'
    a = axx.imshow(
        fits.open(path_equitorial_down)[0].data,
        origin="lower",
        cmap=cmap,
        vmin=28,
        vmax=vmax,
        transform=axx.get_transform(wcs_equitorial_down),
        alpha=alpha,
    )
    xmin, xmax = xlim
    ymin, ymax = ylim

    axx.set_xlim(xmin, xmax)

    axx.set_ylim(ymin, ymax)
    axx.coords[0].set_axislabel(" ")
    axx.coords[1].set_axislabel(" ")

    return fits.open(path_equitorial_down)[0].data


def wise_on_galactic_old(ax, alpha=0.5, cmap="Reds"):
    path = "w40_wise12_continuum.fits"
    import astropy.io.fits as fits
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import matplotlib.tri as tri

    # Read in the fits file
    hdu = fits.open(path)[0]
    wcs = WCS(hdu.header)

    # Get the data
    data = hdu.data

    # fig, ax = plt.subplots()
    # z
    x, y = np.meshgrid(np.arange(4096), np.arange(4096))
    x_flat = x.flatten()
    y_flat = y.flatten()
    intensity_flat = data.flatten()

    ra_dec = wcs.wcs_pix2world(np.transpose([x_flat, y_flat]), 0)
    coord = SkyCoord(ra_dec, frame="icrs", unit="deg")
    x = coord.galactic.l.degree
    y = coord.galactic.b.degree
    x_wise, y_wise = x, y
    intensity_wise = intensity_flat
    interval = 1000
    im = ax.scatter(
        x[::interval],
        y[::interval],
        c=intensity_flat[::interval],
        edgecolors="none",
        marker="s",
        s=300,
        cmap=cmap,
        alpha=alpha,
    )
    # im = ax.scatter(x[::interval],y[::interval],c=intensity_flat[::interval],
    # edgecolors='none',marker='s',s=20,cmap='Greys', alpha=0.7)
    intensity_flat[np.isnan(intensity_flat)] = 0

    ax.set_ylim(3.353333333, 3.686666667)
    ax.set_xlim(28.95666667, 28.62333333)


def wise_on_galactic(ax):
    path = "w40_wise12_continuum.fits"
    wcs = WCS(fits.open(path)[0].header)
    ax.imshow(
        fits.open(path)[0].data,
        origin="lower",
        cmap="cubehelix",
        transform=ax.get_transform(wcs),
        alpha=1,
    )

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)

    ax.coords[0].set_axislabel(" ")
    ax.coords[1].set_axislabel(" ")


dict = {
    "line": ["c2h", "c18o", "cn", "hcn", "hcop", "n2hp"],
    "filename": [
        "w40_c2h_1-8.sdd",
        "w40_c18o_1-9.sdd",
        "w40_cn_1-9.sdd",
        "w40_hcn_1-10.sdd",
        "w40_hcop_1-10.sdd",
        "w40_n2hp_1-8.sdd",
    ],
    "fitsname": [
        "w40_c2h_1-8.fits",
        "w40_c18o_1-9.fits",
        "w40_cn_1-9.fits",
        "w40_hcn_1-10.fits",
        "w40_hcop_1-10.fits",
        "w40_n2hp_1-8.fits",
    ],
    # 'mode':["-70 100",'-50 90','-50 85','-70 100','-40 100','-50 100'],
    "mode": ["-42 58", "-42 58", "-42 58", "-42 58", "-42 58", "-42 58"],
    "window": [
        "-3.2 15.6 -43.7 -25.3",
        "0 15",
        "-22.1 -16.5 1.5 9.9 24.4 31.9",
        "-8 17",
        "-0.8 13.4",
        "-11.5 22.5",
    ],
    "degree": ["5", "1", "4", "3", "3", "2"],
    "contour": [
        [0.6, 0.8, 1],
        [2, 2.5, 3],
        [0.6, 0.8, 1],
        [3, 4, 5],
        [2.5, 3, 3.5],
        [2, 2.5, 3],
    ],
    "v_range": [[0, 15], [0, 15], [20, 35], [0, 15], [0, 15], [0, 15]],
    "color": ["orange", "yellow", "purple", "red", "cyan", "white"],
    "intensity": ["strong", "weak", "weak", "strong", "strong", "weak"],
}
table = pd.DataFrame(dict)


def generate_regional_spectra(line, n_element, vmin, vmax):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    cube = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    # cube = cube.with_spectral_unit(u.K)
    # sub_cube_slab = cube.spectral_slab(vmin*u.km / u.s, vmax *u.km / u.s)
    _, b, _ = cube.world[0, :, 0]
    _, _, l = cube.world[0, 0, :]
    b_range = np.linspace(b[0], b[-1], n_element + 1)
    l_range = np.linspace(l[0], l[-1], n_element + 1)
    # spectra = np.zeros((n_element, n_element))
    spectra = np.empty((n_element, n_element), dtype=object)
    for i in range(n_element):
        for j in range(n_element):
            print(line, i, j)
            lat_range = [b_range[j], b_range[j + 1]] * u.deg
            lon_range = [l_range[i], l_range[i + 1]] * u.deg
            sub_cube = cube.subcube(
                xlo=lon_range[0], xhi=lon_range[1], ylo=lat_range[0], yhi=lat_range[1]
            )
            sub_cube_slab = sub_cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
            ave_spec = [
                np.mean(sub_cube_slab[i]).value for i in range(len(sub_cube_slab))
            ]
            # print(ave_spec)
            # kernel_width = smooth  # Adjust this value to control the smoothing strength
            # kernel = Box1DKernel(kernel_width)
            # smoothed_flux = convolve(ave_spec, kernel)
            spectra[i, j] = ave_spec
    return spectra


def moment_0_map(
    ax, line, vmin, vmax, moment_degree, imshow_min=None, imshow_max=None, opacity=1
):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    cube_c2h = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    sub_cube_slab = cube_c2h.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    moment = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=moment_degree)
    im = ax.imshow(moment.hdu.data, vmax=imshow_max, vmin=imshow_min, alpha=opacity)
    # invert y axis
    ax.invert_yaxis()
    return im


def moment_0_map_sofia(
    ax, vmin, vmax, moment_degree, imshow_min=None, imshow_max=None, opacity=1
):
    u.add_enabled_units(u.def_unit(["K (T_mb)"], represents=u.K))
    cube_c2h = SpectralCube.read("FEEDBACK_W40_CII_OC9N.lmv.fits")
    sub_cube_slab = cube_c2h.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    moment = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=moment_degree)
    im = ax.imshow(moment.hdu.data, vmax=imshow_max, vmin=imshow_min, alpha=opacity)
    # invert y axis
    ax.invert_yaxis()
    return im


def moment_0_contour(ax, line, vmin, vmax, levels, cmap="Greys", alpha=1, line_width=1):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    cube_c2h = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    sub_cube_slab = cube_c2h.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    moment = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=0)
    peak_intensity = max(moment.hdu.data.flatten())
    contour_levels = [peak_intensity * percentage / 100 for percentage in levels]
    contourf = ax.contourf(
        moment.hdu.data,
        levels=contour_levels,
        cmap=cmap,
        transform=ax.get_transform(moment.wcs),
        alpha=1,
    )
    contour2 = ax.contour(
        moment.hdu.data,
        levels=contour_levels,
        cmap=cmap,
        transform=ax.get_transform(moment.wcs),
        alpha=alpha,
        linewidths=line_width,
    )
    return contourf


def moment_0_contour_sofia(ax, vmin, vmax, levels, cmap="Greys", alpha=1, line_width=1):
    u.add_enabled_units(u.def_unit(["K (T_mb)"], represents=u.K))
    cube_c2h = SpectralCube.read("FEEDBACK_W40_CII_OC9N.lmv.fits")
    sub_cube_slab = cube_c2h.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    moment = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=0)
    peak_intensity = np.nanmax(moment.hdu.data.flatten())
    print(peak_intensity)
    contour_levels = [peak_intensity * percentage / 100 for percentage in levels]
    print(contour_levels)
    contourf = ax.contourf(
        moment.hdu.data,
        levels=contour_levels,
        cmap=cmap,
        transform=ax.get_transform(moment.wcs),
        alpha=1,
    )
    contour2 = ax.contour(
        moment.hdu.data,
        levels=contour_levels,
        cmap=cmap,
        transform=ax.get_transform(moment.wcs),
        alpha=alpha,
        linewidths=line_width,
    )
    return contourf


def get_obs_number(row, col, flip=False):
    rows = 61
    cols = 61
    array = np.arange(1, 61**2 + 1, 1).reshape(rows, cols)
    if flip:
        array = np.flipud(array)
    # array = np.flipud(array)
    return array[row, col]


def get_obs_number_phiset(row, col, flip=False):
    rows = 65
    cols = 65
    array = np.arange(1, 65**2 + 1, 1).reshape(rows, cols)
    if flip:
        array = np.flipud(array)
    # array = np.flipud(array)
    array = array[2:63, 2:63]
    return array[row, col]


def get_position(obs_num):
    rows = 61
    cols = 61
    array = np.arange(1, 61**2 + 1, 1).reshape(rows, cols)
    # array = np.flipud(array)
    return np.where(array == obs_num)


def get_position_phiset(obs_num):
    rows = 65
    cols = 65
    array = np.arange(1, 65**2 + 1, 1).reshape(rows, cols)
    array = array[2:63, 2:63]
    # array = np.flipud(array)
    return np.where(array == obs_num)


def get_h2_column_density(obs_num, phiset=False):
    if phiset:
        col, row = get_position_phiset(obs_num)
    else:
        col, row = get_position(obs_num)

    # print(col, row)
    try:
        col, row = col[0], row[0]
        path = "starlink experiment/h2mapingalactic.fits"
        h2_density = fits.open(path)[0].data * 10000
        return h2_density[row, col]
    except:
        return None


def get_h2_column_density2(obs_num, line):
    if line == "hcn" or line == "hcop":
        col, row = get_position_phiset(obs_num)
    else:
        col, row = get_position(obs_num)

    # print(col, row)
    try:
        col, row = col[0], row[0]
        path = "starlink experiment/h2mapingalactic.fits"
        h2_density = fits.open(path)[0].data * 10000
        return h2_density[row, col]
    except:
        return None


import warnings


def moment_0_contour2(ax, line, vmin, vmax, levels, color):
    warnings.filterwarnings("ignore")
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    cube_c2h = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    sub_cube_slab = cube_c2h.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    moment = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=0)
    # peak_intensity = max(moment.hdu.data.flatten())
    # contour_levels = [peak_intensity * percentage / 100 for percentage in levels]
    contour = ax.contour(moment.hdu.data, levels=levels, cmap=color)
    return contour, moment.hdu.data


# smoothing function used in CLASS by default
def hanning_smooth(spectrum, repeat=1):
    for i in range(repeat):
        smoothed_spectrum = np.zeros_like(spectrum, dtype=float)
        N = len(spectrum)
        for i in range(1, N - 1):
            smoothed_spectrum[i] = (
                spectrum[i - 1] / 2 + spectrum[i] + spectrum[i + 1] / 2
            ) / 2
        spectrum = smoothed_spectrum[1::2]

    return spectrum


def halpha_on_galactic(ax, xlim, ylim, cmap="cubehelix", alpha=1, vmax=4769.23):
    path = "ha90823.fits"
    w = WCS(naxis=2)

    w.wcs.crpix = [1.340111109571e03, 1.340295576729e03]
    w.wcs.crval = [2.778604166667e02, -2.072777777778e00]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cdelt = 0.671067 / (3600)
    w.wcs.cdelt = [-cdelt, cdelt]
    ax.imshow(
        fits.open(path)[0].data,
        origin="lower",
        cmap=cmap,
        transform=ax.get_transform(w),
        alpha=alpha,
        vmin=2869.12,
        vmax=4769.23,
    )

    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    ax.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    ax.set_ylim(ymin, ymax)

    ax.coords[0].set_axislabel(" ")
    ax.coords[1].set_axislabel(" ")

    return fits.open(path)[0].data


def spire_on_galactic(
    ax, xlim, ylim, cmap="cubehelix", vmax=8819.889, vmin=49.3664, alpha=1
):
    path = "spire_map.fits"
    wcs = WCS(fits.open(path)[0].header)
    ax.imshow(
        fits.open(path)[1].data,
        origin="lower",
        cmap=cmap,
        transform=ax.get_transform(wcs),
        alpha=alpha,
        vmin=49.3664,
        vmax=vmax,
    )

    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    ax.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    ax.set_ylim(ymin, ymax)

    ax.coords[0].set_axislabel(" ")
    ax.coords[1].set_axislabel(" ")

    return fits.open(path)[1].data


def sofia_cII_on_galactic(ax, vmin=0, vmax=15, cmap="cubehelix", im_min=0, im_max=500):
    u.add_enabled_units(
        u.def_unit(["K (Tmb)"], represents=u.K)
    )  # define the unit K (Tmb) and make it in Kelvin
    cube_c2h = SpectralCube.read(
        "FEEDBACK_W40_CII_OC9N.lmv.fits"
    )  # read the sofia CII spectra cube
    sub_cube_slab = cube_c2h.spectral_slab(
        vmin * u.km / u.s, vmax * u.km / u.s
    )  # limit the spectra cube velocity range to the defined range
    moment0 = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(
        order=0
    )  # get the moment0 map
    ax.imshow(
        moment0.hdu.data,
        origin="lower",
        cmap=cmap,
        transform=ax.get_transform(moment0.wcs),
        vmax=im_max,
        vmin=im_min,
    )  # plot the moment0 map


def h2_column_density_on_galactic(ax, xlim, ylim):
    path = "spire_map.fits"
    wcs = WCS(fits.open(path)[0].header)
    ax.imshow(
        fits.open(path)[1].data,
        origin="lower",
        cmap="cubehelix",
        transform=ax.get_transform(wcs),
        alpha=1,
        # vmin=49.3664,
        vmax=8819.889,
    )

    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    ax.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    ax.set_ylim(ymin, ymax)

    ax.coords[0].set_axislabel(" ")


def plot_irs_sources(
    a, wcs_galactic, xlim, ylim, scale=10, marker="*", opacity=0.5, color="red"
):
    from astropy.table import Table

    table = Table.read("../irs_catalog.txt", format="ascii", delimiter="\t", guess=False)
    coords = SkyCoord(
        table["2MASS R.A. (J2000)"],
        table["2MASS Decl. (J2000)"],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    x, y = wcs_galactic.world_to_pixel(coords)
    a.set_aspect("equal")
    marker_size = (table["T_eff"] / min(table["T_eff"])) * scale
    a.scatter(x, y, marker=marker, s=marker_size, color=color, alpha=opacity)
    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    a.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    a.set_ylim(ymin, ymax)


def plot_dense_cores(a, wcs_galactic, xlim, ylim, s=0.1, color="red", marker="x"):
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs("J/A+A/584/A91")
    dense_cores = catalogs[0]
    coords = SkyCoord(
        dense_cores["RAJ2000"],
        dense_cores["DEJ2000"],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    x, y = wcs_galactic.world_to_pixel(coords)
    a.set_aspect("equal")
    a.scatter(x, y, marker=marker, s=s, color=color, alpha=1)
    xmin, xmax = xlim
    ymin, ymax = ylim
    # axx.set_xlim(0, 60)
    a.set_xlim(xmin, xmax)
    # axx.set_ylim(0, 60)
    a.set_ylim(ymin, ymax)


def generate_spectra(line, vmin, vmax, xlim, ylim, smooth):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    print(fits_name[line])
    cube = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    ymin, ymax = ylim
    xmin, xmax = xlim
    print(xmin, xmax, ymin, ymax)
    sub_cube = cube.subcube(xlo=xmin, xhi=xmax, ylo=ymin, yhi=ymax)
    sub_cube_slab = sub_cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    ave_spec = [np.mean(sub_cube_slab[i]).value for i in range(len(sub_cube_slab))]
    smoothed = hanning_smooth(ave_spec, repeat=smooth)
    return smoothed


def generate_spectra2(line, vmin, vmax, xlim, ylim, smooth):
    u.add_enabled_units(u.def_unit(["K (Ta*)"], represents=u.K))
    print(fits_name[line])
    cube = SpectralCube.read("w40_obs_data/fits_files/" + fits_name[line])
    ymin, ymax = ylim
    xmin, xmax = xlim
    if line == "hcn" and line == "hcop":
        xmin, xmax = xmin + 2, xmax + 2
        ymin, ymax = ymin + 2, ymax + 2
    print(xmin, xmax, ymin, ymax)
    sub_cube = cube.subcube(xlo=xmin, xhi=xmax, ylo=ymin, yhi=ymax)
    sub_cube_slab = sub_cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)
    # ave_spec = [np.mean(sub_cube_slab[i]).value for i in range(len(sub_cube_slab))]
    # smoothed = hanning_smooth(ave_spec, repeat=smooth)
    # return smoothed
    # print(sub_cube_slab.value)
    return sub_cube_slab[:, 0, 0].value


def plot_spectra2(ax, line, xlim, ylim, smooth=1, vmin=0, vmax=15):
    spectra = generate_spectra2(line, vmin, vmax, xlim, ylim, smooth)
    x = np.linspace(vmin, vmax, len(spectra))
    ax.plot(x, spectra, color="black", linewidth=0.5)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("T*A (K)")
    ax.set_title(line)


def plot_spectra(ax, line, vmin, vmax, xlim, ylim, smooth, emission_region):
    spectra = generate_spectra(line, vmin, vmax, xlim, ylim, smooth)
    x = np.linspace(vmin, vmax, len(spectra))
    ax.plot(x, spectra, color="black", linewidth=0.5)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("T_mb (K)")
    # fit continuum on the fly
    # emission_region = '3,4|5,6|7,8|9,10'
    # convert the emission region to a list of set of tuples


def fit_spectra(ax, emission_region):
    emission_region = emission_region.split("|")
    emission_region = [list(map(float, i.split(","))) for i in emission_region]
    spectra = ax.lines[0].get_ydata()
    x = ax.lines[0].get_xdata()
    residual = baseline_fitting(spectra, x, emission_region)
    ax.plot(x, residual, color="red", linewidth=0.5)


def baseline_fitting(spectra, x, emission_region):
    spectrum = Spectrum1D(flux=spectra * u.K, spectral_axis=x * u.km / u.s)
    exclude_regions = []
    for region in emission_region:
        exclude_regions.append(
            SpectralRegion(region[0] * u.km / u.s, region[1] * u.km / u.s)
        )
    g1_fit = fit_generic_continuum(
        spectrum, exclude_regions=[SpectralRegion(2 * u.km / u.s, 10 * u.km / u.s)]
    )
    y_fit = g1_fit(x * u.km / u.s)
    # cont_norm_spec = spectra / y_fit
    residual = spectra - y_fit.value
    return residual


def calculate_fractional_abundance(line, obs_num, fit_results):
    T_bg = 2.73
    n_B = n_B_dict[line]
    v = v_dict[line]
    B_0 = B_0_dict[line]
    g_u = g_u_dict[line]
    A_ij = A_ij_dict[line]

    p1 = fit_results[0]
    p4 = fit_results[3]
    tau_tot = p4  # dimensionless
    fwhm = fit_results[2] * 1000  # km/s to m/s

    T_ex = T_bg + (1 / n_B) * (p1 / p4)
    from scipy.constants import h, k, c
    import numpy as np
    import all_functions

    T_0 = h * v / k
    Q = k * T_ex / (h * B_0) + 1 / 3
    A = (8 * np.pi ** (3 / 2)) / (2 * (np.log(2)) ** (1 / 2))
    B = (Q * v**3) / (c**3 * A_ij * g_u)
    C = (tau_tot * fwhm) / (1 - np.e ** (-T_0 / T_ex))
    column_density = A * B * C
    N_H2 = all_functions.get_h2_column_density2(obs_num, line)
    fractional_abundance = column_density / N_H2
    return fractional_abundance


def save_fit_data(
    line,
    row,
    col,
    obs_num,
    fit_results,
    fit_error,
    n_lines,
    filename,
    base_rms,
    line_rms,
    component=1,
):
    # fit_result = pd.read_csv('fit_result.csv')
    if line == "hcop" or line == "c18o":
        if n_lines == 2:
            print("hello world", fit_results[3:], fit_error[3:])
            save_fit_data(
                line,
                row,
                col,
                obs_num,
                fit_results[3:],
                fit_error[3:],
                1,
                filename,
                base_rms,
                line_rms,
                component=2,
            )
        fit_result = pd.read_csv("fit_result.csv")
        area = fit_results[0] * 1
        position = fit_results[1] * 1
        width = fit_results[2] * 1
        err_area = fit_error[0] * 1
        err_position = fit_error[1] * 1
        err_width = fit_error[2] * 1
        new_row = {
            "line": line,
            "row": row,
            "column": col,
            "obs_num": obs_num,
            "t_ant_tau": None,
            "err_t_ant_tau": None,
            "v_lsr": None,
            "err_v_lsr": None,
            "delta_v": None,
            "err_delta_v": None,
            "tau_main": None,
            "err_tau_main": None,
            "area": area,
            "err_area": err_area,
            "position": position,
            "err_position": err_position,
            "width": width,
            "err_width": err_width,
            "t_peak": None,
            "err_t_peak": None,
            "base_rms": base_rms,
            "line_rms": line_rms,
            "fit_image_link": filename,
            "component": component,
        }
    else:
        if n_lines == 2:
            save_fit_data(
                line,
                row,
                col,
                obs_num,
                fit_results[4:],
                fit_error[4:],
                1,
                filename,
                base_rms,
                line_rms,
                component=2,
            )
        fit_result = pd.read_csv("fit_result.csv")
        t_ant_tau = fit_results[0] * 1
        v_lsr = fit_results[1] * 1
        width = fit_results[2] * 1
        tau_main = fit_results[3] * 1
        err_t_ant_tau = fit_error[0] * 1
        err_v_lsr = fit_error[1] * 1
        err_width = fit_error[2] * 1
        err_tau_main = fit_error[3] * 1
        new_row = {
            "line": line,
            "row": row,
            "column": col,
            "obs_num": obs_num,
            "t_ant_tau": t_ant_tau,
            "err_t_ant_tau": err_t_ant_tau,
            "v_lsr": v_lsr,
            "err_v_lsr": err_v_lsr,
            "delta_v": width,
            "err_delta_v": err_width,
            "tau_main": tau_main,
            "err_tau_main": err_tau_main,
            "area": None,
            "err_area": None,
            "position": None,
            "err_position": None,
            "width": None,
            "err_width": None,
            "t_peak": None,
            "err_t_peak": None,
            "base_rms": base_rms,
            "line_rms": line_rms,
            "fit_image_link": filename,
            "component": component,
        }
    new_row_pd = pd.DataFrame([new_row])
    fit_result = pd.concat([fit_result, new_row_pd], ignore_index=True)
    fit_result.to_csv("fit_result.csv", index=False)
    print("saved")
    return
