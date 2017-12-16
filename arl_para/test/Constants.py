from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.data.polarisation import *
# 常数值
FACETS = 2
PIECE = FACETS * FACETS
NPOL = 4
NCHAN = 5
NX = 256
NY = 256
BEAM = 0
NTIMES = 5
NAN = 3
NBASE = NAN * (NAN - 1) // 2

DX = NX / 4
DY = NY / 4

# 判断是否相等的精度
PRECISION = -5

# 天文相关常量
CELLSIZE = 0.001
CTYPE=["RA---SIN", "DEC--SIN"]
PHASECENTRE = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
POLARISATION_FRAME = PolarisationFrame('linear')