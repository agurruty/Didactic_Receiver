"""A geographic finale from saved estimates; mapping never changes the GNSS solution.
OSM tiles: https://operations.osmfoundation.org/policies/tiles/ (visible viewports only).
Covariance ellipses are constructed in local ENU, then projected with their coordinates.
"""
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from .common import ROOT, INK, TEAL, GOLD, MUTED, finish, write_json
from .position_models import A, E2, ecef, geodetic, enu_matrix

WORLD = 2 * math.pi * A
TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
USER_AGENT = 'DidacticGalileoReceiver/1.0 (scientific GNSS position visualization)'
TTL = 7 * 86400


def mercator(lat, lon, center_lon=None):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if np.any(np.abs(lat) > 85.05112878):
        raise ValueError('Street-map projection supports latitudes within ±85.0511 degrees')
    if center_lon is not None: lon = center_lon + (lon-center_lon+180) % 360-180
    return np.stack((A*np.radians(lon), A*np.arcsinh(np.tan(np.radians(lat)))), axis=-1)


def inverse_mercator(xy):
    xy = np.asarray(xy)
    return np.stack((np.degrees(np.arctan(np.sinh(xy[..., 1]/A))),
                     (np.degrees(xy[..., 0]/A)+180) % 360-180), axis=-1)


def ellipse_geometry(epoch):
    """95% *joint horizontal* ellipse (chi-square, 2 dof), not 1.96 times each sigma."""
    xyz = np.asarray(epoch['state'][:3], dtype=float)
    rotation = enu_matrix(xyz)
    cov = rotation @ np.asarray(epoch['covariance_xyz_clock_m2'])[:3, :3] @ rotation.T
    horizontal = (cov[:2, :2]+cov[:2, :2].T)/2
    values, vectors = np.linalg.eigh(horizontal)
    if not np.all(np.isfinite(values)) or values[0] < -1e-6:
        raise ValueError('Invalid horizontal position covariance')
    radii = np.sqrt(5.991464547107979*np.maximum(values, 0))
    angles = np.linspace(0, 2*math.pi, 181)
    en = vectors @ (radii[:, None]*np.array([np.cos(angles), np.sin(angles)]))
    points = xyz[:, None] + rotation.T @ np.vstack((en, np.zeros(len(angles))))
    latlon = np.array([[math.degrees(v) for v in geodetic(p)[:2]] for p in points.T])
    latlon[-1] = latlon[0]
    bearing = math.degrees(math.atan2(vectors[0, 1], vectors[1, 1])) % 180
    return latlon, radii[::-1], bearing, cov


def viewport(points, latitude, min_width_m, aspect=1.18):
    points = np.asarray(points)
    # Bounds include all accepted estimates and the complete final covariance ellipse.
    low, high = points.min(axis=0), points.max(axis=0)
    width = max(min_width_m/math.cos(math.radians(latitude)),
                (high[0]-low[0])*1.35, (high[1]-low[1])*1.35*aspect)
    center = (high+low)/2
    height = width/aspect
    return [float(center[0]-width/2), float(center[0]+width/2),
            float(center[1]-height/2), float(center[1]+height/2)]


def tile_layout(bounds, pixels=1150, max_tiles=24):
    left, right, bottom, top = bounds
    z = int(np.clip(math.floor(math.log2(WORLD*pixels/(256*(right-left)))), 0, 19))
    while True:
        size = WORLD/(2**z)
        x0, x1 = math.floor((left+WORLD/2)/size), math.floor((right+WORLD/2)/size)
        y0 = max(0, math.floor((WORLD/2-top)/size))
        y1 = min(2**z-1, math.floor((WORLD/2-bottom)/size))
        if (x1-x0+1)*(y1-y0+1) <= max_tiles or z == 0: break
        z -= 1
    if y1 < y0: raise ValueError('Map viewport lies outside the tile projection')
    return z, x0, x1, y0, y1


class TileCache:
    """Bounded requests for displayed viewports; persistent cache with conditional refresh."""
    def __init__(self, root, online=True, deadline_s=35):
        self.root = Path(root); self.online = online
        self.deadline = time.monotonic()+deadline_s

    def get(self, key):
        z, x, y = key
        path = self.root/str(z)/str(x)/f'{y}.png'
        meta_path = path.with_suffix('.json')
        previous = None; meta = {}
        try:
            with Image.open(path) as im:
                if im.size != (256, 256): raise ValueError('Unexpected cached tile dimensions')
                previous = np.array(im.convert('RGB'))
            if meta_path.exists(): meta = json.loads(meta_path.read_text())
            if time.time()-meta.get('checked_at', path.stat().st_mtime) < TTL:
                return previous, 'cache'
        except (OSError, ValueError): previous = None
        if not self.online or time.monotonic() >= self.deadline:
            return previous, 'stale cache' if previous is not None else 'unavailable'
        headers = {'User-Agent': USER_AGENT}
        if meta.get('etag'): headers['If-None-Match'] = meta['etag']
        if meta.get('last_modified'): headers['If-Modified-Since'] = meta['last_modified']
        try:
            request = Request(TILE_URL.format(z=z, x=x, y=y), headers=headers)
            try:
                response = urlopen(request, timeout=max(.1, min(8., self.deadline-time.monotonic())))
            except HTTPError as exc:
                if exc.code == 304 and previous is not None:
                    meta['checked_at'] = time.time(); write_json(meta_path, meta)
                    return previous, 'revalidated cache'
                raise
            with response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000: raise ValueError('Oversized map tile')
                with Image.open(io.BytesIO(raw)) as im:
                    if im.size != (256, 256): raise ValueError('Unexpected map tile dimensions')
                    pixels = np.array(im.convert('RGB'))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                write_json(meta_path, {'checked_at': time.time(), 'etag': response.headers.get('ETag'),
                    'last_modified': response.headers.get('Last-Modified'), 'url': request.full_url})
                return pixels, 'downloaded'
        except (OSError, ValueError) as exc:
            return previous, 'stale cache' if previous is not None else f'unavailable: {type(exc).__name__}'


def backgrounds(bounds_list, cache, online):
    layouts = [tile_layout(bounds, pixels=1150 if i == 0 else 420) for i, bounds in enumerate(bounds_list)]
    keys = sorted({(z, x % (2**z), y) for z, x0, x1, y0, y1 in layouts
                   for x in range(x0, x1+1) for y in range(y0, y1+1)})
    provider = TileCache(cache, online)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tiles = dict(zip(keys, pool.map(provider.get, keys)))
    mosaics = []
    for z, x0, x1, y0, y1 in layouts:
        canvas = np.full(((y1-y0+1)*256, (x1-x0+1)*256, 3), [235, 239, 236], dtype=np.uint8)
        missing = 0
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                pixels, status = tiles[z, x % (2**z), y]
                if pixels is None: missing += 1
                else: canvas[(y-y0)*256:(y-y0+1)*256, (x-x0)*256:(x-x0+1)*256] = pixels
        size = WORLD/(2**z)
        extent = [x0*size-WORLD/2, (x1+1)*size-WORLD/2, WORLD/2-(y1+1)*size, WORLD/2-y0*size]
        mosaics.append((canvas, extent, missing))
    return mosaics, [{'z': z, 'x': x, 'y': y, 'status': tiles[z, x, y][1]} for z, x, y in keys]


def ground_scale(latitude):
    """Local ellipsoidal east metres per Web-Mercator metre for the scale bar."""
    lat = math.radians(latitude)
    return math.cos(lat)/math.sqrt(1-E2*math.sin(lat)**2)


def decorate_map(ax, bounds, latitude, mosaic, source):
    if mosaic is not None:
        pixels, extent, missing = mosaic
        ax.imshow(pixels, extent=extent, origin='upper', interpolation='bilinear', zorder=0)
        if missing:
            ax.text(.5, .87, 'Basemap incomplete · some map tiles unavailable', transform=ax.transAxes,
                    ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.94, edgecolor='none'))
    else:
        ax.set_facecolor('#ebefec')
        ax.text(.5, .87, 'Basemap disabled · geographic coordinates retained', transform=ax.transAxes,
                ha='center', fontsize=9, color=MUTED)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.annotate('N', xy=(.95, .86), xytext=(.95, .95), xycoords='axes fraction',
                ha='center', va='center', color=INK, weight='bold',
                arrowprops=dict(arrowstyle='<|-', color=INK, lw=1.8))
    desired = (bounds[1]-bounds[0])*ground_scale(latitude)*.18
    power = 10**math.floor(math.log10(desired))
    length = max(v*power for v in (1, 2, 5, 10) if v*power <= desired)
    x = bounds[0]+.04*(bounds[1]-bounds[0]); y = bounds[2]+.06*(bounds[3]-bounds[2])
    end = x+length/ground_scale(latitude)
    ax.plot([x, end], [y, y], lw=5, color='white', solid_capstyle='butt', zorder=8)
    ax.plot([x, end], [y, y], lw=2, color=INK, solid_capstyle='butt', zorder=9)
    label = f'{length/1000:g} km' if length >= 1000 else f'{length:g} m'
    ax.annotate(label, (x, y), xytext=(0, 7), textcoords='offset points', fontsize=9, color=INK,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2), zorder=9)
    if source == 'osm':
        ax.text(.99, .01, '© OpenStreetMap contributors', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color=INK, bbox=dict(facecolor='white', alpha=.92, edgecolor='none', pad=3),
                url='https://www.openstreetmap.org/copyright', zorder=10)


def position_map(summary, output, source='osm', width_m=1200., cache=None, reference=None, reference_source=None):
    output = Path(output)
    good = [e for e in summary['epochs'] if e.get('accepted')]
    if not good:
        print('Map: no accepted position; the existing readiness figure explains the missing evidence.')
        return
    last = good[-1]; lat, lon = last['latitude_deg'], last['longitude_deg']
    ellipse_ll, radii, bearing, covariance = ellipse_geometry(last)
    track_ll = np.array([[e['latitude_deg'], e['longitude_deg']] for e in good])
    track = mercator(track_ll[:, 0], track_ll[:, 1], lon)
    ellipse = mercator(ellipse_ll[:, 0], ellipse_ll[:, 1], lon)
    ref = mercator(*reference, center_lon=lon) if reference else None
    points = np.vstack((track, ellipse, [ref] if ref is not None else np.empty((0, 2))))
    local = viewport(points, lat, width_m, aspect=1.36)
    regional = viewport(np.array([[local[0], local[2]], [local[1], local[3]]]), lat, max(40000., width_m*15), aspect=1.77)
    if source == 'osm':
        print('Map: loading OpenStreetMap tiles for the two displayed views (cached between runs).', flush=True)
        mosaics, tile_status = backgrounds([local, regional], cache or ROOT/'runs/.map-cache/osm', True)
    else: mosaics, tile_status = [None, None], []
    fig = plt.figure(figsize=(16, 11), facecolor='#f8f9f6')
    fig.text(.045, .955, '05  /  THE SIGNAL FINDS A PLACE', color=TEAL, fontsize=11, weight='bold', va='top')
    fig.text(.045, .915, 'From the antenna to the map', color=INK, fontsize=28, weight='bold', va='top')
    fig.text(.045, .864, f"Recorded Galileo {summary.get('mode','e1').upper()} · estimated antenna location · final accepted epoch", fontsize=12, color=MUTED)
    ax = fig.add_axes([.045, .185, .605, .645])
    decorate_map(ax, local, lat, mosaics[0], source)
    ax.add_patch(Polygon(ellipse, closed=True, facecolor=GOLD, alpha=.17, edgecolor='none', zorder=3))
    ax.plot(*ellipse.T, color=GOLD, lw=2.2, zorder=4)
    ax.scatter(*track.T, color=TEAL, s=13, alpha=.65,
               linewidths=.3, edgecolors='white', zorder=5)
    ax.scatter(*track[-1], s=210, facecolors='white', edgecolors=TEAL, linewidths=2.4, zorder=7)
    ax.scatter(*track[-1], s=40, color=TEAL, zorder=8)
    ax.annotate('FINAL ESTIMATE', track[-1], xytext=(25, -33), textcoords='offset points',
                fontsize=9, weight='bold', color=TEAL, zorder=10,
                bbox=dict(boxstyle='round,pad=.5', facecolor='white', edgecolor='none', alpha=.95),
                arrowprops=dict(arrowstyle='-', color=TEAL, connectionstyle='angle,angleA=0,angleB=90,rad=5'))
    actual_error = None
    if ref is not None:
        ref_xyz = ecef(math.radians(reference[0]), math.radians(reference[1]), last['ellipsoidal_height_m'])
        error_enu = enu_matrix(ref_xyz) @ (np.asarray(last['state'][:3])-ref_xyz)
        actual_error = float(np.linalg.norm(error_enu[:2]))
        ax.scatter(*ref, marker='+', s=180, linewidths=2.5, color='#ae4667', zorder=9)
        ax.plot([ref[0], track[-1, 0]], [ref[1], track[-1, 1]], '--', color='#ae4667', lw=1.2, zorder=6)
        ax.annotate('REFERENCE', ref, xytext=(10, 10), textcoords='offset points', color='#ae4667', fontsize=9, zorder=10)
    # An explicit legend separates repeatability from the model-based uncertainty region.
    ax.plot([], [], 'o', ms=6, color=TEAL, label=f'{len(good)} accepted epoch estimates')
    ax.plot([], [], color=GOLD, lw=2.2, label='Conditional 95% horizontal ellipse')
    ax.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white', edgecolor='none', framealpha=.94)
    side = fig.add_axes([.69, .45, .275, .38]); side.set_axis_off(); side.set(xlim=(0, 1), ylim=(0, 1))
    side.text(0, 1, 'FINAL ESTIMATE', color=TEAL, fontsize=10, weight='bold', va='top')
    side.text(0, .90, f'{abs(lat):.7f}° {"N" if lat>=0 else "S"}', fontsize=23, weight='bold', va='top')
    side.text(0, .79, f'{abs(lon):.7f}° {"E" if lon>=0 else "W"}', fontsize=23, weight='bold', va='top')
    side.text(0, .66, f'Ellipsoidal height   {last["ellipsoidal_height_m"]:.1f} m', fontsize=11)
    side.text(0, .58, f'{len(last["observations"])} satellites   ·   PDOP {last["pdop"]:.2f}', fontsize=11)
    side.axhline(.51, color='#d9e0dc', lw=1)
    side.text(0, .43, 'CONDITIONAL 95% HORIZONTAL UNCERTAINTY', color=TEAL, fontsize=9, weight='bold')
    side.text(0, .32, f'{radii[0]:.0f} × {radii[1]:.0f} m', fontsize=25, weight='bold', color=INK)
    side.text(0, .23, f'Semi-axes · major axis {bearing:.0f}° east of north', fontsize=10, color=MUTED)
    side.text(0, .12, f'Vertical model uncertainty   ±{math.sqrt(covariance[2,2]):.0f} m (1σ)', fontsize=10, color=MUTED)
    side.text(0, .01, f'Horizontal offset from reference: {actual_error:.1f} m' if actual_error is not None
              else 'Actual position error: unknown\nNo surveyed reference supplied', va='bottom', fontsize=9, color='#ae4667' if ref is not None else MUTED)
    inset = fig.add_axes([.69, .185, .275, .225])
    decorate_map(inset, regional, lat, mosaics[1], source)
    inset.add_patch(Rectangle((local[0], local[2]), local[1]-local[0], local[3]-local[2],
                             fill=False, edgecolor=TEAL, lw=1.4, zorder=3))
    inset.scatter(*track[-1], color=TEAL, edgecolors='white', linewidths=1.4, s=70, zorder=4)
    inset.set_title('WIDER AREA  /  detail shown at left', loc='left', fontsize=10, color=TEAL, pad=8)
    uncorrected = 'uncorrected' in summary['corrections']['ionosphere'].lower()
    status = 'PROVISIONAL · ionosphere uncorrected' if uncorrected else 'MODEL-DEPENDENT POSITION ESTIMATE'
    redundancy = 'Four satellites: no residual redundancy.' if last['residual_dof'] == 0 else f'{last["residual_dof"]} residual degrees of freedom; not certified integrity.'
    fig.text(.045, .135, status, color='#9b651c', fontsize=11, weight='bold')
    fig.text(.045, .106, 'The ellipse describes the assumed range-error model, not a measured accuracy bound. '+redundancy, fontsize=10, color=INK)
    fig.text(.045, .082, 'Atmosphere, multipath and uncalibrated hardware biases may move the entire cluster outside this ellipse.', fontsize=10, color=MUTED)
    fig.text(.045, .047, 'Projection: Web Mercator · ellipse transformed from local ENU · scale bar corrected at the estimate latitude', fontsize=9, color=MUTED)
    link = f'https://www.openstreetmap.org/?mlat={lat:.8f}&mlon={lon:.8f}#map=17/{lat:.8f}/{lon:.8f}'
    fig.text(.965, .047, 'Explore this location on OpenStreetMap ↗', ha='right', fontsize=9, color=TEAL, url=link)
    finish(fig, output/'05_position_map')
    write_json(output/'position_map.json', dict(source=source, projection='EPSG:3857',
        latitude_deg=lat, longitude_deg=lon, final_receive_time_s=last['receive_time_s'], accepted_epochs=len(good),
        ellipse_semi_axes_m=radii.tolist(), ellipse_major_bearing_deg=bearing,
        uncertainty='Conditional 95% joint horizontal covariance ellipse; not a measured error bound',
        reference_lat_lon=reference, reference_source=reference_source, horizontal_reference_offset_m=actual_error,
        map_link=link, view_bounds_mercator_m=[local, regional], tiles=tile_status))
    # A geographic artifact lets the same real measurements be inspected in a GIS.
    features = [dict(type='Feature', properties={'name':'Final estimate'}, geometry=dict(type='Point',coordinates=[lon,lat])),
        dict(type='Feature', properties={'name':'Conditional 95% horizontal ellipse','semi_axes_m':radii.tolist()},
             geometry=dict(type='Polygon',coordinates=[ellipse_ll[:, ::-1].tolist()]))]
    features.extend(dict(type='Feature', properties={'receive_time_s':e['receive_time_s']},
                         geometry=dict(type='Point',coordinates=[e['longitude_deg'],e['latitude_deg']])) for e in good)
    if reference:
        features.append(dict(type='Feature', properties={'name':'Supplied reference','source':reference_source},
                             geometry=dict(type='Point',coordinates=[reference[1],reference[0]])))
    write_json(output/'position_map.geojson',dict(type='FeatureCollection',features=features))
    (output/'open_in_openstreetmap.html').write_text('<!doctype html><meta charset="utf-8"><title>Receiver location</title>'
        f'<p><a href="{link}">Explore the final GNSS estimate on OpenStreetMap</a></p>'
        '<p>The marker is an estimate, not surveyed truth. See 05_position_map.png for conditional uncertainty.</p>',encoding='utf8')
    print(f'Map: {link}')


def add_map_arguments(parser):
    parser.add_argument('--map-source', choices=('osm', 'none'), default='osm', help='Street basemap for the final figure; none avoids network requests.')
    parser.add_argument('--map-width', type=float, default=1200., help='Minimum detailed map width [ground metres]; expands to fit estimates and uncertainty.')
    parser.add_argument('--map-cache', type=Path, help='Persistent OSM tile cache; default runs/.map-cache/osm.')
    parser.add_argument('--map-reference', type=float, nargs=2, metavar=('LAT', 'LON'), help='Optional independently known reference latitude/longitude [degrees].')
    parser.add_argument('--map-reference-source', help='Source of the supplied reference coordinate; required with --map-reference.')


def validate_map_arguments(args):
    if not math.isfinite(args.map_width) or not 50 <= args.map_width <= 200000:
        raise ValueError('--map-width must be 50..200000 ground metres')
    if args.map_reference is not None:
        lat, lon = args.map_reference
        if not all(map(math.isfinite, (lat, lon))) or abs(lat)>85.05112878 or abs(lon)>180:
            raise ValueError('Map reference must be a finite latitude/longitude inside Web Mercator coverage')
        if not args.map_reference_source:
            raise ValueError('--map-reference requires --map-reference-source')


def render_map(summary, output, args):
    """A basemap failure must not invalidate a saved GNSS solution."""
    try:
        position_map(summary, output, source=args.map_source, width_m=args.map_width,
                     cache=args.map_cache, reference=args.map_reference, reference_source=args.map_reference_source)
    except (OSError, ValueError) as exc:
        write_json(Path(output)/'position_map.json',dict(status='map unavailable',reason=str(exc)))
        print(f'Map unavailable: {exc}. Numerical position results are retained.')
        return False
    return bool(summary.get('accepted_epochs'))
