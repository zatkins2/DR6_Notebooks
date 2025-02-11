import numpy as np
from pixell import utils, enmap
from scipy import ndimage
import warnings

def get_tmap_tiles(tmap: enmap.ndmap, 
                   grid_deg: float, 
                   zeromap: enmap.ndmap, 
                   id=None
                   ):
    tile_map = tiles_t_quick(tmap, grid_deg, id=id)
    tile_map[np.where(zeromap == 0.0)] = 0.0
    return tile_map


def get_medrat(snr: enmap, tiledmap):
    """
    gets median ratio for map renormalization given tiles

    Args:
         snr: snr map
         tiledmap: tiles from tmap to get ratio from

    Returns:
        median ratio for each tile
    """
    from scipy.stats import norm
    t = tiledmap.astype(int)
    med0 = norm.ppf(0.75) 
    medians = ndimage.median(snr**2, labels=t, index=np.arange(np.max(t + 1)))
    median_map = medians[t]
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med_ratio = med0 / median_map**0.5
    med_ratio[np.where(t == 0)] = 0.0
    return med_ratio


def get_non_zero_pix(tmap):
    """
    Find non-zero pixel at around half of dec range

    Args:
        tmap: ndmap of time map

    Returns:
        dec_pix, ra_pix: pixel of non-zero pixel
    """
    dec_pix_0 = int(tmap.shape[0] / 2)
    ra_pix = 0
    ra_inds = range(int(tmap.shape[1] / (utils.degree / utils.arcmin)))
    dec_inds_pos = range(int(tmap.shape[0] / (2 * utils.degree / utils.arcmin)))
    dec_inds_neg = range(
        0, 0 - int(tmap.shape[0] / (2 * utils.degree / utils.arcmin)), -1
    )
    dec_inds = list(dec_inds_pos) + list(dec_inds_neg)
    for j in dec_inds:  # adding 0.5 degree increment each time in dec
        dec_pix = dec_pix_0 + int((j) * utils.degree / utils.arcmin)
        for i in ra_inds:
            if (
                tmap[dec_pix, int((i + 1) * utils.degree / utils.arcmin)] > 0.0
            ):  # adding 0.5 degree increment each time in ra
                ra_pix = int((i + 1) * utils.degree / utils.arcmin)
                break
            else:
                continue
        break
    return dec_pix, ra_pix


def time_shift_in_ra(tmap):
    """
    Gets time difference for 1 degree shift in ra at the center

    Args:
        tmap: ndmap of time map
        shift_deg: shift in ra in degree

    Returns:
        t_shift: time shift in seconds
    """

    dec_pix0, ra_pix0 = get_non_zero_pix(tmap)
    t0 = tmap[dec_pix0, ra_pix0]
    t1 = 0.0
    t_shift = 0.0

    for i in range(8):
        # try to find another point for t2 that is not in empty area
        t1 = tmap[
            dec_pix0, ra_pix0 + int((i + 2) * 0.25 / np.abs(tmap.wcs.wcs.cdelt[0]))
        ]  # try to get an ra_shift increasing from 0.25 by 0.25 degree increment
        if t1 > 0:
            # rescale the time shift to 1degree in ra
            t_shift = np.abs(t1 - t0) / ((i + 2) * 0.25)
            break
        else:
            continue

    return t_shift


def time_shift_in_ra_new(tmap):
    """
    Gets time difference for 1 degree shift in ra at the center

    Args:
        tmap: ndmap of time map
        shift_deg: shift in ra in degree

    Returns:
        t_shift: time shift in seconds
    """
    dec_pix_0 = int(tmap.shape[0] / 2)
    ra_pix = 0
    res = np.abs(tmap.wcs.wcs.cdelt[0])
    ra_inds = range(int(tmap.shape[1] * 2 * res))
    dec_inds_pos = range(int(tmap.shape[0] * res))
    dec_inds_neg = range(0, 0 - int(tmap.shape[0] * res), -1)
    dec_inds = list(dec_inds_pos) + list(dec_inds_neg)
    t_shift = 0
    for j in dec_inds:  # adding 0.5 degree increment each time in dec
        dec_pix = dec_pix_0 + int((j) * 0.5 / res)
        for i in ra_inds:  # adding 0.5 degree increment each time in ra
            ra_pix = int(i * 0.5 / res)
            ra_pix_shift = int((i + 1) * 0.5 / res)
            if (
                tmap[dec_pix, ra_pix] != 0.0 and tmap[dec_pix, ra_pix_shift] != 0.0
            ):  # adding 0.5 degree increment each time in ra
                t_shift = (
                    np.abs(tmap[dec_pix, ra_pix_shift] - tmap[dec_pix, ra_pix]) / 0.5
                )
                if t_shift > 100:
                    break  # take 4-6 min to drift across one array
        else:
            continue
        break
    return t_shift


def get_decs(tmap, grid_deg):
    """
    Returns a bunch of decs (in pixel) given a resolution

    Args:
        tmap: ndmap of time map
        grid_deg: resolution in degree

    Returns:
        decs_pix: array of decs in pixel
    """
    indices = np.where((tmap != 0).any(axis=1))[0]
    dec_pix_min = np.min(indices) - 1
    dec_pix_max = np.max(indices) + 1
    if dec_pix_min > 0:
        dec_pix_min -= 1
    if dec_pix_max < tmap.shape[0]:
        dec_pix_max += 1
    grid_pix = int(grid_deg / np.abs(tmap.wcs.wcs.cdelt[0]))
    offsets_dec = int((dec_pix_max - dec_pix_min) / grid_pix) + 1
    decs_pix = [dec_pix_min + i * grid_pix for i in range(offsets_dec)]
    decs_pix.append(dec_pix_max)
    decs_pix = np.array(decs_pix)
    return decs_pix


def tiles_t(tmap, grid_deg):
    """takes tmap as imput and return a tilemap with pixels having same ind number beloing to the same time

    Args:
        tmap: ndmap of time map
        grid_deg: resolution in degree

    Returns:
        mask_poly: ndmap of tile map with pixels labeled with tile number
    """

    t_max = np.max(tmap)
    t_shift_1deg = time_shift_in_ra_new(tmap)
    t_shift = t_shift_1deg * grid_deg
    if t_shift == 0.0:
        raise ValueError("did not find proper non zero pixel to measure t_shift")
    t_offsets = int(t_max / t_shift) + 1
    decs_pix = get_decs(tmap, grid_deg)
    mask_poly = enmap.zeros(tmap.shape, tmap.wcs)
    index = 1
    for i in range(t_offsets):
        if i != t_offsets - 1:
            t1 = i * t_shift
            t2 = (i + 1) * t_shift
        else:
            t1 = i * t_shift
            t2 = t_max + 60
        mask_time = np.zeros((tmap.shape[0], tmap.shape[1]))
        mask_time[np.where((tmap >= t1) & (tmap < t2))] = 1
        for j in range(decs_pix.shape[0] - 1):
            dec1_pix = int(decs_pix[j])
            dec2_pix = int(decs_pix[j + 1])
            mask_dec = np.zeros((tmap.shape[0], tmap.shape[1]))
            mask_dec[dec1_pix:dec2_pix, :] = 1
            mask_final = mask_dec * mask_time
            mask_poly[np.where(mask_final != 0)] = index
            index += 1
    mask_poly[np.where(tmap == 0)] = 0
    # reindex tiles so that indices are continuous
    mask_reind = enmap.zeros(tmap.shape, tmap.wcs)
    index = 0
    for i in range(1, int(np.max(mask_poly)) + 1):
        if i in mask_poly:
            mask_reind[np.where(mask_poly == i)] = index + 1
            index += 1
    return mask_reind


def tiles_t_quick(tmap, grid_deg, id=None):
    """takes tmap as input and return a tilemap with pixels having same ind number belonging to the same time

    Args:
        tmap: ndmap of time map
        grid_deg: resolution in degree

    Returns:
        mask_poly: ndmap of tile map with pixels labeled with tile number
    """
    t_max = np.nanmax(tmap)
    t_shift_1deg = time_shift_in_ra_new(tmap)
    t_shift = t_shift_1deg * grid_deg
    if t_shift == 0.0:
        if id is not None:
            print(
                f"Warning: did not find proper non zero pixel to measure t_shift for {id}"
            )
        raise ValueError("did not find proper non zero pixel to measure t_shift")
    t_offsets = int(t_max / t_shift) + 1
    decs_pix = get_decs(tmap, grid_deg)
    mask_ra = enmap.zeros(tmap.shape, tmap.wcs)
    mask_dec = enmap.zeros(tmap.shape, tmap.wcs)
    index_ra = 0
    index_dec = 0
    for i in range(t_offsets):
        if i != t_offsets - 1:
            t1 = i * t_shift
            t2 = (i + 1) * t_shift
        else:
            t1 = i * t_shift
            t2 = t_max + 60
        index_ra += 1
        mask_ra[np.where((tmap >= t1) & (tmap < t2))] = index_ra

    for j in range(decs_pix.shape[0] - 1):
        dec1_pix = int(decs_pix[j])
        dec2_pix = int(decs_pix[j + 1])
        index_dec += 1
        mask_dec[dec1_pix:dec2_pix, :] = index_dec

    mask_poly = (mask_ra - 1) * index_dec + mask_dec
    mask_poly[np.where(tmap == 0)] = 0
    return mask_poly


def apply_tiles(imap, tiles):
    """breaks imap into tiles and returns a list of tiles

    Args:
        imap: ndmap of imap
        tiles: ndmap of tiles

        Returns:
            tiles_list: list of tiles
    """

    tiles_list = []
    for i in range(1, int(np.max(tiles)) + 1):
        mask = np.zeros(tiles.shape)
        mask[np.where(tiles == i)] = 1
        tile = imap * mask
        tiles_list.append(tile)
    return tiles_list


def root_median_square(imap, tiles):
    """returns a list of root median square of tiles

    Args:
        imap: ndmap of imap
        tiles: list of tiles

    Returns:
        medians: list of medians
    """
    medians = ndimage.median(
        imap**2.0, labels=tiles, index=np.arange(np.max(tiles) + 1)
    )
    return medians**0.5


def get_median(imap, tiles, dec, ra):
    """returns median of the tile that contains ra, dec

    Args:
        imap: ndmap of imap
        tiles: list of tiles
        ra: ra in degree
        dec: dec in degree

    Returns:
        median: median of the tile
    """

    cors_pix = imap.sky2pix(np.array([np.deg2rad(dec), np.deg2rad(ra)]))
    dec_pix = cors_pix[0]
    ra_pix = cors_pix[1]
    tile_num = tiles[int(dec_pix), int(ra_pix)]
    tile = imap[np.where(tiles == tile_num)]
    median = np.sqrt(np.median(tile**2))
    return median


def get_tile_center(tmap, grid_deg, dec, ra):
    """returns a tile that's centered at ra, dec

    Args:
        tmap:time map associated with the imap
        grid_deg: side length of the tile in degree
        ra: ra in degree
        dec: dec in degree
    Returns:
        mask: ndmap of tile map with a tile labeled 1 and centered at ra,dec, 0 elsewhere
    """
    dec_rad = dec * utils.degree
    ra_rad = ra * utils.degree
    grid_half = grid_deg * 0.5
    time = tmap.at([dec_rad, ra_rad])
    time_ndeg = tmap.at([dec_rad, ra_rad + 1 * utils.degree])
    n = 1
    for i in [-5, -4, -3, -2, -1, 2, 3, 4, 5]:
        if time_ndeg != 0.0:
            break
        else:
            time_ndeg = tmap.at([dec_rad, ra_rad + i * utils.degree])
            n = i
    time_shift = np.abs((time_ndeg - time) / n) * grid_half
    map_box = enmap.corners(tmap.shape, tmap.wcs, npoint=10, corner=True)
    map_box_deg = np.rad2deg(map_box)
    map_dec_min = np.min(map_box[:, 0])  # in rad
    map_dec_max = np.max(map_box[:, 0])
    dec_max = np.min([map_dec_max, dec_rad + grid_half * utils.degree])
    dec_min = np.max(
        [map_dec_min, dec_rad - grid_half * utils.degree]
    )  # in case it's out of range
    dec_max_pix = int(tmap.sky2pix(np.array([dec_max, 0]))[0])
    dec_min_pix = int(tmap.sky2pix(np.array([dec_min, 0]))[0])
    mask_dec = np.zeros((tmap.shape[0], tmap.shape[1]))
    mask_dec[dec_min_pix:dec_max_pix, :] = 1
    mask_time = np.zeros((tmap.shape[0], tmap.shape[1]))
    mask_time[np.where((tmap >= time - time_shift) & (tmap < time + time_shift))] = 1
    mask_final = mask_dec * mask_time
    mask_final[np.where(tmap == 0)] = 0.0
    return mask_final


def get_center_median(imap, mask):
    """
    returns median of centered tile

    Args:
        imap: ndmap of imap
        mask: ndmap of centered tile

    Returns:
        median: median of the tile
    """

    center = imap[np.where(mask == 1)]
    median = np.sqrt(np.median(center**2))
    return median


def median_of_tiles(medians):
    """
    returns median of medians excluding zeros

    Args:
        medians: list of medians

    Returns:
        median: median of medians
    """

    medians = np.array(medians)
    medians = medians[np.where(medians != 0)]
    median = np.median(medians)
    return median


def mean_of_tiles(medians):
    """
    returns mean of medians excluding zeros

    Args:
        medians: list of medians

    Returns:
        mean: mean of medians
    """

    medians = np.array(medians)
    medians = medians[np.where(medians != 0)]
    mean = np.mean(medians)
    return mean
