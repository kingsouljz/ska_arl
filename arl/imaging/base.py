"""
Functions that aid fourier transform processing. These are built on top of the core
functions in arl.fourier_transforms.

The measurement equation for a sufficently narrow field of view interferometer is:

.. math::

    V(u,v,w) =\\int I(l,m) e^{-2 \\pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

.. math::

    V(u,v,w) =\\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approachs for dealing with the wide-field problem where the
extra phase term in the Fourier transform cannot be ignored.
"""

import collections
import logging
from typing import List, Union

import numpy
from astropy import constants as constants
from astropy import units as units
from astropy import wcs
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.data_models import Visibility, BlockVisibility, Image, Skycomponent, assert_same_chan_pol
from arl.data.parameters import get_parameter
from arl.data.polarisation import convert_pol_frame, PolarisationFrame
from arl.fourier_transforms.convolutional_gridding import convolutional_grid, \
    convolutional_degrid
from arl.fourier_transforms.fft_support import fft, ifft, pad_mid, extract_mid
from arl.image.operations import create_image_from_array
from arl.imaging.params import get_frequency_map, get_polarisation_map, get_uvw_map, get_kernel_list
from arl.util.coordinate_support import simulate_point, skycoord_to_lmn
from arl.visibility.base import copy_visibility, phaserotate_visibility
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility

log = logging.getLogger(__name__)


def shift_vis_to_image(vis: Visibility, im: Image, tangent: bool = True, inverse: bool = False) -> Visibility:
    """Shift visibility to the FFT phase centre of the image in place

    :param vis: Visibility data
    :param im: Image model used to determine phase centre
    :param tangent: Is the shift purely on the tangent plane True|False
    :param inverse: Do the inverse operation True|False
    :return: visibility with phase shift applied and phasecentre updated

    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    nchan, npol, ny, nx = im.data.shape
    
    # Convert the FFT definition of the phase center to world coordinates (1 relative)
    # This is the only place in ARL where the relationship between the image and visibility
    # frames is defined.
    
    image_phasecentre = pixel_to_skycoord(nx // 2, ny // 2, im.wcs, origin=1)
    
    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        if inverse:
            log.debug("shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                      "%s" % (image_phasecentre, vis.phasecentre))
        else:
            log.debug("shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s" %
                      (vis.phasecentre, image_phasecentre))
        vis = phaserotate_visibility(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre
    
    assert type(vis) is Visibility, "after phase_rotation, vis is not a Visibility"
    
    return vis


def normalize_sumwt(im: Image, sumwt) -> Image:
    """Normalize out the sum of weights

    :param im: Image, im.data has shape [nchan, npol, ny, nx]
    :param sumwt: Sum of weights [nchan, npol]
    """
    nchan, npol, _, _ = im.data.shape
    assert nchan == sumwt.shape[0]
    assert npol == sumwt.shape[1]
    for chan in range(nchan):
        for pol in range(npol):
            if sumwt[chan, pol] > 0.0:
                im.data[chan, pol, :, :] /= sumwt[chan, pol]
            else:
                im.data[chan, pol, :, :] = 0.0
    return im


def predict_2d_base(vis: Visibility, model: Image, **kwargs) -> Visibility:
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """
    if type(vis) is not Visibility:
        avis = coalesce_visibility(vis, **kwargs)
    else:
        avis = vis
    _, _, ny, nx = model.data.shape
    # print(model.shape)
    spectral_mode, vfrequencymap = get_frequency_map(avis, model) # 可以并行
    polarisation_mode, vpolarisationmap = get_polarisation_map(avis, model, **kwargs) # 可以并行
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(avis, model, **kwargs) # 可以并行
    kernel_name, gcf, vkernellist = get_kernel_list(avis, model, **kwargs)
    uvgrid = fft((pad_mid(model.data, int(round(padding * nx))) * gcf).astype(dtype=complex))
    avis.data['vis'] = convolutional_degrid(vkernellist, avis.data['vis'].shape, uvgrid,
                                            vuvwmap, vfrequencymap, vpolarisationmap)
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(avis, model, tangent=True, inverse=True)

    if type(vis) is not Visibility:
        return decoalesce_visibility(svis)
    else:
        return svis


def predict_2d(vis: Visibility, im: Image, **kwargs) -> Visibility:
    """ Predict using convolutional degridding and w projection
    
    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """
    log.debug("predict_2d: predict using 2d transform")
    return predict_2d_base(vis, im, **kwargs)


def invert_2d_base(vis: Visibility, im: Image, dopsf: bool = False, normalize: bool = True, **kwargs) \
        -> (Image, numpy.ndarray):
    """ Invert using 2D convolution function, including w projection optionally

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. . Any shifting needed is performed here.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image

    """
    if type(vis) is not Visibility:
        svis = coalesce_visibility(vis, **kwargs)
    else:
        svis = copy_visibility(vis)

    if dopsf:
        svis.data['vis'] = numpy.ones_like(svis.data['vis'])

    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)
    
    nchan, npol, ny, nx = im.data.shape
    
    spectral_mode, vfrequencymap = get_frequency_map(svis, im)
    polarisation_mode, vpolarisationmap = get_polarisation_map(svis, im, **kwargs)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map(svis, im, **kwargs)
    kernel_name, gcf, vkernellist = get_kernel_list(svis, im, **kwargs)
    
    # Optionally pad to control aliasing
    imgridpad = numpy.zeros([nchan, npol, int(round(padding * ny)), int(round(padding * nx))], dtype='complex')
    imgridpad, sumwt = convolutional_grid(vkernellist, imgridpad, svis.data['vis'],
                                          svis.data['imaging_weight'],
                                          vuvwmap,
                                          vfrequencymap, vpolarisationmap)
    
    # Fourier transform the padded grid to image, multiply by the gridding correction
    # function, and extract the unpadded inner part.
    
    # Normalise weights for consistency with transform
    sumwt /= float(padding * int(round(padding * nx)) * ny)
    
    imaginary = get_parameter(kwargs, "imaginary", False)
    if imaginary:
        log.debug("invert_2d_base: retaining imaginary part of dirty image")
        result = extract_mid(ifft(imgridpad) * gcf, npixel=nx)
        resultreal = create_image_from_array(result.real, im.wcs)
        resultimag = create_image_from_array(result.imag, im.wcs)
        if normalize:
            resultreal = normalize_sumwt(resultreal, sumwt)
            resultimag = normalize_sumwt(resultimag, sumwt)
        return resultreal, sumwt, resultimag
    else:
        result = extract_mid(numpy.real(ifft(imgridpad)) * gcf, npixel=nx)
        resultimage = create_image_from_array(result, im.wcs)
        if normalize:
            resultimage = normalize_sumwt(resultimage, sumwt)
        return resultimage, sumwt


def invert_2d(vis: Visibility, im: Image, dopsf=False, normalize=True, **kwargs)  -> (Image, numpy.ndarray):
    """ Invert using prolate spheroidal gridding function

    Use the image im as a template. Do PSF in a separate call.

    Note that the image is not normalised but the sum of the weights. This is for ease of use in partitioning.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :return: resulting image[nchan, npol, ny, nx], sum of weights[nchan, npol]

    """
    log.debug("invert_2d: inverting using 2d transform")
    kwargs['kernel'] = get_parameter(kwargs, "kernel", '2d')
    return invert_2d_base(vis, im, dopsf, normalize=normalize, **kwargs)

def predict_skycomponent_blockvisibility(vis: BlockVisibility,
                                         sc: Union[Skycomponent, List[Skycomponent]], **kwargs) -> BlockVisibility:
    """Predict the visibility from a Skycomponent, add to existing visibility, for BlockVisibility

    :param vis: BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :param spectral_mode: {mfs|channel} (channel)
    :return: BlockVisibility
    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    nchan = vis.nchan
    npol = vis.npol
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    k = vis.frequency / constants.c.to('m/s').value
    
    for comp in sc:
        
        assert_same_chan_pol(vis, comp)
        
        flux = comp.flux
        if comp.polarisation_frame != vis.polarisation_frame:
            flux = convert_pol_frame(flux, comp.polarisation_frame, vis.polarisation_frame)
        
        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        for chan in range(nchan):
            phasor = simulate_point(vis.uvw * k[chan], l, m)
            for pol in range(npol):
                vis.data['vis'][..., chan, pol] += flux[chan, pol] * phasor[...]
    
    return vis


def predict_skycomponent_visibility(vis: Visibility, sc: Union[Skycomponent, List[Skycomponent]]) -> Visibility:
    """Predict the visibility from a Skycomponent, add to existing visibility, for Visibility

    :param vis: Visibility
    :param sc: Skycomponent or list of SkyComponents
    :return: Visibility
    """
    assert type(vis) is Visibility, "vis is not a Visibility: %r" % vis
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    
    _, ichan = list(get_frequency_map(vis, None))

    npol = vis.polarisation_frame.npol
    
    for comp in sc:
        
        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        phasor = simulate_point(vis.uvw, l, m)
        for ivis in range(vis.nvis):
            for pol in range(npol):
                vis.data['vis'][ivis, pol] += comp.flux[ichan[ivis], pol] * phasor[ivis]
            
            # coords = phasor, ichan
            # for pol in range(npol):
            #     vis.data['vis'][:,pol] += [comp.flux[ic, pol] * p for p, ic in zip(*coords)]
    
    return vis


def create_image_from_visibility(vis: Visibility, **kwargs) -> Image:
    """Make an empty image from params and Visibility

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channel_bandwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param nchan: Number of image channels (Default is 1 -> MFS)
    :return: image
    """
    assert type(vis) is Visibility or type(vis) is BlockVisibility, \
        "vis is not a Visibility or a BlockVisibility: %r" % (vis)
    
    log.info("create_image_from_visibility: Parsing parameters to get definition of WCS")
    
    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)
    
    # Spectral processing options
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)
    
    frequency = get_parameter(kwargs, "frequency", vis.frequency)
    inchan = get_parameter(kwargs, "nchan", vnchan)
    reffrequency = frequency[0] * units.Hz
    channel_bandwidth = get_parameter(kwargs, "channel_bandwidth", vis.channel_bandwidth[0]) * units.Hz
    
    if (inchan == vnchan) and vnchan > 1:
        log.info(
            "create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
            % (inchan, imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining single channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif inchan > 1 and vnchan > 1:
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining multi-channel MFS Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channel_bandwidth.value) > 0.0, "Channel width must be non-zero for mfs mode"
        log.info("create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
                 "and bandwidth %s"
                 % (imagecentre, reffrequency, channel_bandwidth))
    else:
        raise ValueError("create_image_from_visibility: unknown spectral mode ")
    
    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    uvmax = numpy.max((numpy.abs(vis.data['uvw'][:, 0:1])))
    if type(vis) is BlockVisibility:
        uvmax *= numpy.max(frequency) / constants.c.to('m/s').value
    log.info("create_image_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.info("create_image_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.info("create_image_from_visibility: Cellsize          = %f radians, %f degrees" % (cellsize,
                                                                                           cellsize * 180.0 / numpy.pi))
    if cellsize > criticalcellsize:
        log.info("create_image_from_visibility: Resetting cellsize %f radians to criticalcellsize %f radians" % (
            cellsize, criticalcellsize))
        cellsize = criticalcellsize
    pol_frame = get_parameter(kwargs, "polarisation_frame", PolarisationFrame("stokesI"))
    inpol = pol_frame.npol
    
    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth.to(units.Hz).value]
    # The numpy definition of the phase centre of an FFT is n // 2 (0 - rel) so that's what we use for
    # the reference pixel. We have to use 0 rel everywhere.
    w.wcs.crpix = [npixel // 2, npixel // 2, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, reffrequency.to(units.Hz).value]
    w.naxis = 4
    
    w.wcs.radesys = get_parameter(kwargs, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(kwargs, 'equinox', 2000.0)
    
    return create_image_from_array(numpy.zeros(shape), wcs=w, polarisation_frame=pol_frame)


def residual_image(vis: Visibility, model: Image, invert_residual=invert_2d, predict_residual=predict_2d,
                   **kwargs) -> Image:
    """Calculate residual image and visibility

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param invert: invert to be used (default invert_2d)
    :param predict: predict to be used (default predict_2d)
    :return: residual visibility, residual image, sum of weights
    """
    visres = copy_visibility(vis, zero=True)
    visres = predict_residual(visres, model, **kwargs)
    visres.data['vis'] = vis.data['vis'] - visres.data['vis']
    dirty, sumwt = invert_residual(visres, model, dopsf=False, **kwargs)
    return visres, dirty, sumwt